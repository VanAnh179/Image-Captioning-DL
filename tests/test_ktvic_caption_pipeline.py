import tempfile
import unicodedata
import unittest
from pathlib import Path

from PIL import Image

from scripts.prepare_ktvic_dataset import split_images
from src.data.vietnamese_caption_batch import (
    BOS, EOS, PAD, UNK, CaptionVocabulary, make_caption_batch, normalize_caption,
)


class CaptionPipelineTests(unittest.TestCase):
    def test_normalization_shift_padding_and_decode(self):
        text = unicodedata.normalize("NFD", "  MỘT\tem,  bé!  ")
        self.assertEqual(normalize_caption(text), "một em bé")
        vocabulary = CaptionVocabulary.from_training_captions([text])
        with tempfile.TemporaryDirectory() as directory:
            Image.new("RGB", (8, 5), "blue").save(Path(directory) / "sample.png")
            samples = [(1, "sample.png", text), (2, "sample.png", "lạ")]
            batch = make_caption_batch(samples, vocabulary, directory)
        words = vocabulary.encode("một em bé")
        self.assertEqual(batch["decoder_input"].tolist(), [[BOS, *words], [BOS, UNK, PAD, PAD]])
        self.assertEqual(batch["labels"].tolist(), [[*words, EOS], [UNK, EOS, PAD, PAD]])
        self.assertEqual(batch["valid_token_mask"].sum().item(), 6)
        self.assertEqual(tuple(batch["images"].shape), (2, 3, 224, 224))
        self.assertEqual(vocabulary.decode([BOS, *words, EOS, UNK]), "một em bé")

    def test_duplicate_pixels_stay_together_and_test_wins(self):
        train = {i: {"image_id": i, "pixel_sha256": str(i), "captions": ["a"]} for i in range(30)}
        train[9]["pixel_sha256"] = train[0]["pixel_sha256"]
        test = {30: {"image_id": 30, "pixel_sha256": "1", "captions": ["b"]}}
        splits, excluded, groups = split_images(train, test, 42)
        self.assertEqual(excluded[0]["train_id"], 1)
        self.assertEqual(groups, [[0, 9]])
        self.assertEqual(splits["test"], list(test.values()))
        self.assertEqual(len(splits["val"]), 3)
        self.assertEqual(len(splits["train"]), 26)
        shuffled = dict(reversed(list(train.items())))
        self.assertEqual(split_images(shuffled, test, 42)[0], splits)
        # A resized copy of ID 9 links its exact-pixel family to the official test.
        filtered, removed, _ = split_images(train, test, 42, [[9, 30]])
        self.assertEqual([entry["train_id"] for entry in removed], [0, 1, 9])
        self.assertEqual(filtered["test"], list(test.values()))
