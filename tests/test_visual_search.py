import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from src.retrieval.caption_search import SearchInputError
from src.retrieval.visual_search import (FORMAT_VERSION, SOURCE_FILE, TEXT_MODEL, TEXT_MODEL_FILES, TEXT_REVISION,
                                         VisualImageSearch, VisualSearchIndex, check_text_model_dir)

DIGEST = "a" * 64


def vector(first, second=0.0):
    values = torch.zeros(384)
    values[:2] = torch.tensor([first, second])
    return values


def image_index():
    return VisualSearchIndex(torch.tensor([30, 20, 10]), torch.stack((vector(0, 1), vector(1), vector(1))),
                             vector(0), "val", DIGEST)


def caption_index():
    records = [{"image_id": image_id, "image_path": f"val/{image_id}.jpg", "caption": f"caption {image_id}"}
               for image_id in (10, 20, 30)]
    return SimpleNamespace(records=records, metadata={"candidate_split": "val", "val_manifest_sha256": DIGEST})


class VisualSearchTests(unittest.TestCase):
    def setUp(self):
        self.encoder = SimpleNamespace(encode=lambda texts: torch.stack([vector(1) for _ in texts]))

    def test_ranks_same_score_by_id_and_displays_original_captions(self):
        search = VisualImageSearch(image_index(), caption_index(), self.encoder)
        results = search.search("một người đi xe", k=2)
        self.assertEqual([hit["image_id"] for hit in results], [10, 20])
        self.assertEqual([hit["rank"] for hit in results], [1, 2])
        self.assertEqual([hit["caption"] for hit in results], ["caption 10", "caption 20"])

    def test_val_and_test_pool_is_checked_against_the_pool_digest(self):
        from src.retrieval.caption_search import pool_digest

        digest = pool_digest(["a" * 64, "b" * 64])
        index = VisualSearchIndex(torch.tensor([30, 20, 10]), torch.stack((vector(0, 1), vector(1), vector(1))),
                                  vector(0), "val+test", digest)
        captions = caption_index()
        captions.metadata = {"candidate_split": "val+test", "val+test_manifest_sha256": digest}
        VisualImageSearch(index, captions, self.encoder)
        captions.metadata["val+test_manifest_sha256"] = pool_digest(["b" * 64, "a" * 64])  # other split order
        with self.assertRaisesRegex(ValueError, "manifest"):
            VisualImageSearch(index, captions, self.encoder)

    def test_validates_manifest_and_candidate_ids(self):
        captions = caption_index()
        captions.metadata["val_manifest_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "manifest"):
            VisualImageSearch(image_index(), captions, self.encoder)
        captions = caption_index()
        captions.records.pop()
        with self.assertRaisesRegex(ValueError, "ID"):
            VisualImageSearch(image_index(), captions, self.encoder)

    def test_rejects_empty_or_non_text_queries(self):
        search = VisualImageSearch(image_index(), caption_index(), self.encoder)
        for query, code in (("...?!", "empty_query"), (5, "invalid_query")):
            with self.assertRaises(SearchInputError) as captured:
                search.search(query)
            self.assertEqual(captured.exception.code, code)

    def test_load_validates_revision_and_scores_centered_query(self):
        data = {
            "format_version": FORMAT_VERSION, "text_model": TEXT_MODEL, "text_revision": TEXT_REVISION,
            "split": "val", "manifest_sha256": DIGEST, "image_ids": torch.tensor([1, 2]),
            "vectors": torch.stack((vector(0, 1), vector(1))), "text_center": vector(1),
        }
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "index.pt"
            torch.save(data, path)
            index = VisualSearchIndex.load(path)
            torch.testing.assert_close(index.scores(vector(1, 1).unsqueeze(0)), torch.tensor([[1.0, 0.0]]))
            torch.save({**data, "text_revision": "old"}, path)
            with self.assertRaisesRegex(ValueError, "phiên bản"):
                VisualSearchIndex.load(path)


class TextModelFolderTests(unittest.TestCase):
    def make_folder(self, folder, revision=TEXT_REVISION):
        for name in TEXT_MODEL_FILES:
            (Path(folder) / name).write_bytes(b"x")
        (Path(folder) / SOURCE_FILE).write_text(json.dumps({"model": TEXT_MODEL, "revision": revision}), encoding="utf-8")

    def test_complete_folder_is_accepted(self):
        with tempfile.TemporaryDirectory() as folder:
            self.make_folder(folder)
            check_text_model_dir(folder)

    def test_missing_file_names_the_file_and_the_export_command(self):
        with tempfile.TemporaryDirectory() as folder:
            self.make_folder(folder)
            (Path(folder) / "model.safetensors").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "model.safetensors.*export_text_model"):
                check_text_model_dir(folder)

    def test_other_revision_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            self.make_folder(folder, revision="old")
            with self.assertRaisesRegex(ValueError, "không phải bản"):
                check_text_model_dir(folder)


class FeatureMixTests(unittest.TestCase):
    def test_one_feature_keeps_the_original_normalization_and_two_features_give_unit_vectors(self):
        from torch.nn import functional as F

        from scripts.build_visual_search_index import normalize_parts

        torch.manual_seed(0)
        first, second = torch.randn(6, 8), torch.randn(6, 5)
        state_first = (first.mean(0), first.std(0).clamp_min(0.025))
        state_second = (second.mean(0), second.std(0).clamp_min(1e-3))
        single = normalize_parts([first], [state_first])
        torch.testing.assert_close(single, F.normalize((first - state_first[0]) / state_first[1], dim=1))
        both = normalize_parts([first, second], [state_first, state_second])
        self.assertEqual(both.shape, (6, 13))
        torch.testing.assert_close(both.norm(dim=1), torch.ones(6))
