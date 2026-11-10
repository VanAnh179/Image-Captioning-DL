import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from src.data.region_features import (
    FEATURE_DIM, NUM_REGIONS, FrozenResNet50Grid, RegionFeatureStore, load_image, pixel_sha256,
)
from src.data.vietnamese_caption_batch import CaptionVocabulary, make_caption_batch


def tiny_store():
    features = torch.arange(3, dtype=torch.float16).view(3, 1, 1).expand(3, NUM_REGIONS, FEATURE_DIM).clone()
    return RegionFeatureStore({
        "image_ids": torch.tensor([7, 3, 9]), "features": features,
        "image_paths": ["a.png", "b.png", "c.png"], "pixel_sha256": ["ha", "hb", "hc"], "meta": {},
    })


class RegionFeatureTests(unittest.TestCase):
    def test_image_tensor_matches_shared_batch_pipeline(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "x.png"
            Image.effect_noise((31, 17), 60).convert("RGB").save(path)
            tensor, digest = load_image(path)
            batch = make_caption_batch([(1, "x.png", "a")], CaptionVocabulary.from_training_captions(["a"]), directory)
            with Image.open(path) as image:
                self.assertEqual(digest, pixel_sha256(image.convert("RGB")))
        self.assertTrue(torch.equal(tensor, batch["images"][0]))

    def test_encoder_gives_49_regions_and_stays_frozen(self):
        encoder = FrozenResNet50Grid(pretrained=False)
        encoder.train()
        self.assertFalse(encoder.training)
        self.assertFalse(any(p.requires_grad for p in encoder.parameters()))
        out = encoder(torch.rand(2, 3, 224, 224))
        self.assertEqual(tuple(out.shape), (2, NUM_REGIONS, FEATURE_DIM))
        self.assertFalse(out.requires_grad)

    def test_lookup_follows_requested_id_order_and_reports_missing_ids(self):
        store = tiny_store()
        self.assertEqual(store.get_batch([9, 7])[:, 0, 0].tolist(), [2.0, 0.0])
        self.assertEqual(store.get_batch(torch.tensor([3]))[:, 0, 0].tolist(), [1.0])
        with self.assertRaisesRegex(KeyError, "123"):
            store.get_batch([7, 123])
        self.assertEqual(store.trace(3)["image_path"], "b.png")

    def test_tensor_traces_back_to_its_id(self):
        store = tiny_store()
        for image_id in (7, 3, 9):
            noisy = store.get_batch([image_id])[0] + 0.01
            self.assertEqual(store.nearest_id(noisy)[0], image_id)


if __name__ == "__main__":
    unittest.main()
