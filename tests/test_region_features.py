import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from scripts.verify_region_features import verify_split
from src.data.region_features import (
    FEATURE_DIM, FORMAT_VERSION, NUM_REGIONS, WEIGHTS_NAME,
    FrozenResNet50Grid, RegionFeatureStore, load_image, pixel_sha256,
)
from src.data.vietnamese_caption_batch import CaptionVocabulary, make_caption_batch


def tiny_store():
    features = torch.arange(3, dtype=torch.float16).view(3, 1, 1).expand(3, NUM_REGIONS, FEATURE_DIM).clone()
    return RegionFeatureStore({
        "image_ids": torch.tensor([7, 3, 9]), "features": features,
        "image_paths": ["a.png", "b.png", "c.png"], "pixel_sha256": ["ha", "hb", "hc"], "meta": {},
    })


class RegionFeatureTests(unittest.TestCase):
    def test_verifier_detects_changed_manifest_ids_and_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows, features = [], []
            for image_id in (7, 3):
                image_path = f"{image_id}.png"
                Image.new("RGB", (8, 8), (image_id,) * 3).save(root / image_path)
                tensor, digest = load_image(root / image_path)
                rows.append({"image_id": image_id, "image_path": image_path, "pixel_sha256": digest})
                features.append(tensor[0, 0, 0].expand(NUM_REGIONS, FEATURE_DIM).clone())
            manifest = root / "val_manifest.jsonl"
            manifest.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
            (root / "val_ids.json").write_text("[7, 3]", encoding="utf-8")
            torch.save({"image_ids": torch.tensor([7, 3]), "features": torch.stack(features).half(),
                        "image_paths": [r["image_path"] for r in rows],
                        "pixel_sha256": [r["pixel_sha256"] for r in rows],
                        "meta": {"format_version": FORMAT_VERSION, "split": "val", "weights": WEIGHTS_NAME,
                                 "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}},
                       root / "resnet50_7x7_val.pt")

            def encoder(images):
                return images[:, 0, 0, 0].reshape(-1, 1, 1).expand(-1, NUM_REGIONS, FEATURE_DIM)

            report, errors, _ = verify_split("val", root, root, root, encoder, "cpu", 2, 42, 2)
            self.assertEqual(errors, [])
            self.assertEqual(report["traced_ok"], 2)
            rows[0]["image_id"] = 99
            manifest.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
            _, errors, _ = verify_split("val", root, root, root, encoder, "cpu", 2, 42, 2)
            self.assertTrue(any("ID/thứ tự trong manifest" in error for error in errors))
            self.assertTrue(any("hash manifest" in error for error in errors))

    def test_store_rejects_misaligned_metadata_wrong_shapes_and_duplicate_ids(self):
        store = tiny_store()
        payload = {"image_ids": store.image_ids, "features": store.features,
                   "image_paths": store.image_paths, "pixel_sha256": store.pixel_sha256, "meta": {}}
        for change in ({"image_paths": ["a.png"]}, {"pixel_sha256": []},
                       {"features": store.features[:, :1]}, {"image_ids": torch.tensor([7, 7, 9])}):
            with self.assertRaises(ValueError):
                RegionFeatureStore({**payload, **change})

    def test_saved_store_round_trip_keeps_ids_and_tensors(self):
        store = tiny_store()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resnet50_7x7_val.pt"
            torch.save({"image_ids": store.image_ids, "features": store.features,
                        "image_paths": store.image_paths, "pixel_sha256": store.pixel_sha256, "meta": {}}, path)
            loaded = RegionFeatureStore.load(directory, "val")
            self.assertTrue(torch.equal(loaded.get_batch([9, 7]), store.get_batch([9, 7])))
            self.assertEqual(loaded.trace(3), store.trace(3))

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
