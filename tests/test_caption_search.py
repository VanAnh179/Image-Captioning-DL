import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.retrieval.caption_search import CaptionSearchIndex


class CaptionSearchTests(unittest.TestCase):
    def setUp(self):
        self.train = [{"image_id": 10, "captions": ["a a mèo", "a chó"]}]
        self.candidates = [{"image_id": 2, "image_path": "2.jpg", "captions": ["bí mật val"]},
                           {"image_id": 1, "image_path": "1.jpg", "captions": ["bí mật val"]}]
        self.predictions = [(2, "mèo"), (1, "chó")]
        self.index = CaptionSearchIndex.build(self.train, self.predictions, self.candidates)

    def test_idf_uses_only_train_documents_not_candidate_references(self):
        self.assertNotIn("bí", self.index.vocabulary)
        self.assertNotIn("mật", self.index.vocabulary)
        np.testing.assert_allclose(self.index.idf[self.index.vocabulary.index("a")], 1.0)
        np.testing.assert_allclose(self.index.idf[self.index.vocabulary.index("mèo")], np.log(3 / 2) + 1)
        self.assertEqual(self.index.records[0]["caption"], "chó")
        self.assertEqual(self.index.metadata["train_documents"], 2)

    def test_search_ranks_correct_caption_and_breaks_ties_by_numeric_id(self):
        result = self.index.search("MÈO!!!", image_root="data/raw/ktvic")
        self.assertEqual([r["image_id"] for r in result], [2, 1])
        self.assertAlmostEqual(result[0]["score"], 1.0)
        self.assertTrue(Path(result[0]["image_path"]).is_absolute())
        tied = CaptionSearchIndex.build(self.train, [(2, "mèo"), (1, "mèo")], self.candidates)
        self.assertEqual([r["image_id"] for r in tied.search("mèo")], [1, 2])

    def test_empty_unknown_queries_and_invalid_predictions_are_rejected(self):
        for query in ("   ,.!", "không xuất hiện"):
            with self.assertRaises(ValueError):
                self.index.search(query)
        with self.assertRaises(ValueError):
            self.index.search("mèo", k=0)
        for bad in ([(1, "mèo")], [(1, "mèo"), (1, "chó"), (2, "mèo")], [(1, "mèo"), (99, "chó")]):
            with self.assertRaises(ValueError):
                CaptionSearchIndex.build(self.train, bad, self.candidates)

    def test_save_and_reload_preserve_ranking_without_pickle(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "index.npz"
            self.index.save(path)
            loaded = CaptionSearchIndex.load(path)
            self.assertEqual(loaded.search("mèo chó"), self.index.search("mèo chó"))
            self.assertEqual(loaded.metadata, self.index.metadata)
            with np.load(path, allow_pickle=False) as archive:
                self.assertEqual(archive["vectors"].shape, (2, 3))


if __name__ == "__main__":
    unittest.main()
