import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from src.retrieval.caption_search import SearchInputError
from src.retrieval.visual_search import FORMAT_VERSION, TEXT_MODEL, TEXT_REVISION, VisualImageSearch, VisualSearchIndex

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
