import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import torch

from scripts.compare_caption_models import compare_models
from src.evaluation.caption_scorer import score_captions


class CaptionComparisonTests(unittest.TestCase):
    def test_comparison_uses_same_ids_and_reports_a_minus_a0(self):
        references = {1: ["người đàn ông đang chạy trên đường"],
                      2: ["một con chó đang đứng trên bãi cỏ"],
                      3: ["cô gái đang đi xe đạp"]}
        a = [(i, refs[0]) for i, refs in references.items()]
        a0 = [(1, a[1][1]), (2, a[2][1]), (3, a[0][1])]
        config = {"vocab_size": 30, "d_model": 16, "hidden_size": 24, "attention_size": 12, "dropout": 0.1}
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            root = Path(folder)
            (root / "val_manifest.jsonl").write_text("\n".join(json.dumps({"image_id": i, "image_path": f"{i}.jpg", "captions": refs}) for i, refs in references.items()), encoding="utf-8")
            for name, predictions in (("A", a), ("A0", a0)):
                directory = root / name
                directory.mkdir()
                model_config = {**config, **({"attention_mode": "uniform"} if name == "A0" else {})}
                metrics = {"config": model_config, "settings": {"seed": 42}, "data_signature": {"val": "same"},
                           "best_epoch": 2, "scores": score_captions(predictions, references)}
                (directory / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
                (directory / "val_predictions.json").write_text(json.dumps([{"image_id": i, "caption": c} for i, c in predictions]), encoding="utf-8")
                torch.save({"model_name": name, "config": model_config}, directory / "best.pt")
            report = compare_models(root / "A", root / "A0", root)
            self.assertEqual(report["changed_captions"], 3)
            self.assertGreater(report["delta_a_minus_a0"]["CIDEr"], 0)
            self.assertEqual({r["image_id"] for r in report["examples"]}, set(references))
            self.assertEqual(report["attention_mode"], "uniform")
            path = root / "A0/metrics.json"
            changed = json.loads(path.read_text(encoding="utf-8"))
            changed["settings"]["seed"] = 1
            path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "settings"):
                compare_models(root / "A", root / "A0", root)


if __name__ == "__main__":
    unittest.main()
