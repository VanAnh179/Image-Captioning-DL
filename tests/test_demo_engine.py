import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.data.region_features import FrozenResNet50Grid
from src.demo.app import vi_score
from src.demo.engine import DemoEngine, check_index_matches_checkpoint, first_existing
from src.retrieval.caption_search import CaptionSearchIndex, SearchInputError

TRAIN = [{"image_id": 1, "captions": ["xe máy đỗ bên đường", "kệ hàng bày nhiều hàng hoá", "con mèo ngủ trên ghế",
                                      "người đàn ông đứng trên đường", "nhiều xe máy trên đường"]}]
CANDIDATES = [{"image_id": 10, "image_path": "train-images/a.jpg", "captions": ["x"] * 5},
              {"image_id": 11, "image_path": "train-images/b.jpg", "captions": ["x"] * 5},
              {"image_id": 12, "image_path": "train-images/c.jpg", "captions": ["x"] * 5}]
PREDICTIONS = [(10, "xe máy đỗ bên đường"), (11, "kệ hàng bày nhiều hàng hoá"), (12, "con mèo ngủ trên ghế")]


def tiny_engine():
    engine = object.__new__(DemoEngine)  # skip loading the checkpoint and ResNet-50
    engine.index = CaptionSearchIndex.build(TRAIN, PREDICTIONS, CANDIDATES)
    engine.image_root = Path("root")
    return engine


class DemoEngineTests(unittest.TestCase):
    def test_search_returns_ranked_images_with_caption_and_matching_words(self):
        hits = tiny_engine().search("kệ hàng", 5)
        self.assertEqual([hit["image_id"] for hit in hits], [11])  # only images with a positive score
        self.assertEqual(hits[0]["rank"], 1)
        self.assertEqual(hits[0]["caption"], "kệ hàng bày nhiều hàng hoá")
        self.assertEqual(set(hits[0]["matched_words"]), {"kệ", "hàng"})
        self.assertTrue(str(hits[0]["image_path"]).endswith("b.jpg"))

    def test_search_orders_by_score_and_limits_to_k(self):
        hits = tiny_engine().search("xe máy trên đường", 2)
        self.assertEqual(len(hits), 2)
        self.assertGreaterEqual(hits[0]["score"], hits[1]["score"])
        self.assertEqual(hits[0]["image_id"], 10)

    def test_empty_and_unknown_queries_are_reported_not_crashed(self):
        engine = tiny_engine()
        with self.assertRaises(SearchInputError) as empty:
            engine.search("   ", 5)
        self.assertEqual(empty.exception.code, "empty_query")
        with self.assertRaises(SearchInputError) as unknown:
            engine.search("zzz qqq", 5)
        self.assertEqual(unknown.exception.code, "unknown_terms")

    def test_index_must_hold_the_captions_of_the_checkpoint(self):
        index = tiny_engine().index
        same = [{"image_id": i, "caption": c} for i, c in PREDICTIONS]
        self.assertTrue(check_index_matches_checkpoint(index, {"validation_predictions": same}))
        other = same[:2] + [{"image_id": 12, "caption": "khác"}]
        self.assertFalse(check_index_matches_checkpoint(index, {"validation_predictions": other}))
        self.assertIsNone(check_index_matches_checkpoint(index, {}))

    def test_index_may_hold_extra_images_but_must_not_miss_or_change_validation_captions(self):
        index = tiny_engine().index
        validation = [{"image_id": 10, "caption": "xe máy đỗ bên đường"}, {"image_id": 11, "caption": "kệ hàng bày nhiều hàng hoá"}]
        self.assertTrue(check_index_matches_checkpoint(index, {"validation_predictions": validation}))  # image 12 is extra
        absent = validation + [{"image_id": 99, "caption": "x"}]
        self.assertFalse(check_index_matches_checkpoint(index, {"validation_predictions": absent}))

    def test_default_checkpoint_lookup_and_score_format(self):
        with tempfile.TemporaryDirectory() as folder:
            present = Path(folder) / "best.pt"
            present.write_bytes(b"x")
            self.assertEqual(first_existing([str(Path(folder) / "missing.pt"), str(present)]), present)
            with self.assertRaisesRegex(FileNotFoundError, "--checkpoint"):
                first_existing([str(Path(folder) / "missing.pt")])
        self.assertEqual(vi_score(0.694144), "0,694")

    def test_offline_encoder_stops_when_weights_are_missing(self):
        with (tempfile.TemporaryDirectory() as folder,
              patch("src.data.region_features.torch.hub.get_dir", return_value=folder),
              patch("src.data.region_features.torchvision.models.resnet50") as model):
            with self.assertRaisesRegex(FileNotFoundError, "resnet50-0676ba61.pth"):
                FrozenResNet50Grid(local_only=True)
            model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
