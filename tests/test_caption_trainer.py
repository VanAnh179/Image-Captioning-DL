import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.nn import functional as F

from src.data.cached_caption_dataset import CachedCaptionDataset
from src.data.region_features import RegionFeatureStore
from src.data.vietnamese_caption_batch import PAD, CaptionVocabulary
from src.evaluation.caption_scorer import ScorerInputError, score_captions
from src.training.caption_trainer import atomic_save, build_model, fit, logits_from, token_loss
from scripts.export_caption_results import export_results


def tiny_data():
    torch.manual_seed(123)
    captions = ["một con chó", "hai bé gái", "một người", "bé gái đi học"]
    vocabulary = CaptionVocabulary.from_training_captions(captions)
    rows = [{"image_id": index, "captions": [caption]} for index, caption in enumerate(captions)]
    store = RegionFeatureStore({"image_ids": torch.arange(4), "features": torch.randn(4, 49, 2048),
                                "image_paths": [f"{i}.jpg" for i in range(4)], "pixel_sha256": [str(i) for i in range(4)], "meta": {}})
    return CachedCaptionDataset(rows, vocabulary, store)


class CaptionTrainerTests(unittest.TestCase):
    def setUp(self):
        self.data = tiny_data()
        self.configs = {
            "A": {"vocab_size": len(self.data.vocabulary.tokens), "d_model": 16, "hidden_size": 16, "attention_size": 16, "dropout": 0.2},
            "B": {"vocab_size": len(self.data.vocabulary.tokens), "d_model": 16, "num_heads": 4, "num_layers": 1, "d_ff": 32, "dropout": 0.2, "max_len": 16},
        }
        self.settings = {"batch_size": 3, "learning_rate": 0.001, "weight_decay": 0.01,
                         "selection_metric": "val_loss", "max_len": 8, "seed": 42, "purpose": "test"}

    def assert_nested_equal(self, left, right):
        if torch.is_tensor(left):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_nested_equal(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_nested_equal(a, b)
        else:
            self.assertEqual(left, right)

    def test_resume_matches_uninterrupted_parameters_optimizer_rng_and_losses_for_both_models(self):
        for name, config in self.configs.items():
            with self.subTest(model=name), tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
                root = Path(folder)
                fit(name, config, self.data, self.data, root / "full", self.settings, 3)
                fit(name, config, self.data, self.data, root / "resumed", self.settings, 1)
                fit(name, config, self.data, self.data, root / "resumed", self.settings, 3, resume=root / "resumed/last.pt")
                full = torch.load(root / "full/last.pt", weights_only=True)
                resumed = torch.load(root / "resumed/last.pt", weights_only=True)
                for key in ("model_state", "optimizer_state", "rng_state", "epoch", "global_step"):
                    self.assert_nested_equal(full[key], resumed[key])
                self.assertEqual([row["train"]["loss"] for row in full["history"]],
                                 [row["train"]["loss"] for row in resumed["history"]])
                self.assertTrue(resumed["optimizer_state"]["state"])
                with self.assertRaisesRegex(ValueError, "settings"):
                    fit(name, config, self.data, self.data, root / "resumed", {**self.settings, "batch_size": 2},
                        4, resume=root / "resumed/last.pt")

    def test_added_padding_preserves_loss_and_valid_token_count_for_both_models(self):
        batch = self.data.collate([self.data[i] for i in range(3)])
        for name, config in self.configs.items():
            model = build_model(name, config).eval()
            with torch.no_grad():
                original = logits_from(model(batch["regions"], batch["decoder_input"]))
                padded = logits_from(model(batch["regions"], F.pad(batch["decoder_input"], (0, 5), value=PAD)))
                loss, count = token_loss(original, batch["labels"])
                padded_loss, padded_count = token_loss(padded, F.pad(batch["labels"], (0, 5), value=PAD))
            torch.testing.assert_close(loss, padded_loss, atol=1e-6, rtol=1e-6)
            self.assertEqual(count, padded_count)
            manual = F.cross_entropy(original[batch["labels"].ne(PAD)], batch["labels"][batch["labels"].ne(PAD)])
            torch.testing.assert_close(loss, manual)
        with self.assertRaisesRegex(ValueError, "non-PAD"):
            token_loss(torch.randn(1, 2, 10), torch.zeros(1, 2, dtype=torch.long))

    def test_best_checkpoint_survives_worse_epochs_and_is_repaired_on_resume(self):
        def stats(loss):
            return {"loss": loss, "tokens": 10, "batches": 1, "seconds": 0}

        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            root = Path(folder)
            with patch("src.training.caption_trainer.run_epoch", side_effect=[stats(2), stats(1), stats(1), stats(3)]):
                fit("A", self.configs["A"], self.data, self.data, root, self.settings, 2)
            self.assertEqual(torch.load(root / "best.pt", weights_only=True)["epoch"], 1)
            self.assertEqual(torch.load(root / "last.pt", weights_only=True)["epoch"], 2)
            (root / "best.pt").unlink()
            (root / "history.json").write_text("broken", encoding="utf-8")
            fit("A", self.configs["A"], self.data, self.data, root, self.settings, 2, resume=root / "last.pt")
            self.assertEqual(torch.load(root / "best.pt", weights_only=True)["epoch"], 1)

    def test_interrupted_save_preserves_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "last.pt"
            atomic_save({"epoch": 1}, path)
            with patch("src.training.caption_trainer.torch.save", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    atomic_save({"epoch": 2}, path)
            self.assertEqual(torch.load(path, weights_only=True)["epoch"], 1)
            self.assertEqual(list(Path(folder).glob("*.tmp")), [])

    def test_cider_selection_exports_best_captions_and_stops_after_patience(self):
        stats = {"loss": 1.0, "tokens": 10, "batches": 1, "seconds": 0.01}
        scores = [({"scores": {"CIDEr": value}}, [{"image_id": 0, "caption": caption}])
                  for value, caption in ((10, "best"), (5, "worse"), (4, "last"))]
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            root = Path(folder)
            with patch("src.training.caption_trainer.run_epoch", return_value=stats), \
                    patch("src.training.caption_trainer.evaluate_captions", side_effect=scores):
                fit("A", self.configs["A"], self.data, self.data, root,
                    {**self.settings, "selection_metric": "cider", "patience": 2}, 10)
            last = torch.load(root / "last.pt", weights_only=True)
            best = torch.load(root / "best.pt", weights_only=True)
            self.assertEqual(last["epoch"], 3)
            self.assertEqual(best["epoch"], 1)
            export_results(root)
            self.assertIn('"best"', (root / "val_predictions.json").read_text())
            self.assertTrue((root / "loss_curve.png").is_file())

    def test_training_scorer_accepts_eos_only_without_weakening_default_validation(self):
        predictions = [(0, ""), (1, "hai bé gái")]
        references = {0: ["một con chó"], 1: ["hai bé gái"]}
        with self.assertRaises(ScorerInputError):
            score_captions(predictions, references)
        result = score_captions(predictions, references, allow_empty=True)
        self.assertEqual(result["num_images"], 2)


if __name__ == "__main__":
    unittest.main()
