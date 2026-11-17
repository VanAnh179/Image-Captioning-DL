import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModel

from scripts.check_clip_pipeline import manual_clip_preprocess
from src.data.clip_features import CLIP_CACHE_PREFIX, FrozenCLIPVisionGrid, load_clip_image
from src.data.region_features import FEATURE_DIM, NUM_REGIONS, RegionFeatureStore, feature_path, pixel_sha256
from src.data.vietnamese_caption_batch import EOS, PAD
from src.models.transformer_decoder import TransformerCaptionDecoder
from src.training.caption_trainer import generate_tokens

VOCAB = 50
MEAN, STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)


def tiny_clip():
    torch.manual_seed(0)
    config = CLIPVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                              image_size=224, patch_size=32)
    return CLIPVisionModel(config)


def clip_processor():  # the values stored in openai/clip-vit-base-patch32/preprocessor_config.json, without the network
    return CLIPImageProcessor(size={"shortest_edge": 224}, crop_size={"height": 224, "width": 224}, resample=3,
                              image_mean=list(MEAN), image_std=list(STD))


class ClipEncoderTests(unittest.TestCase):
    def test_grid_has_49_regions_without_class_token_and_is_frozen(self):
        encoder = FrozenCLIPVisionGrid(vision_model=tiny_clip())
        pixels = torch.randn(3, 3, 224, 224)
        regions = encoder(pixels)
        self.assertEqual(regions.shape, (3, NUM_REGIONS, 32))
        full = encoder.model(pixel_values=pixels).last_hidden_state
        self.assertEqual(full.shape[1], NUM_REGIONS + 1)
        self.assertTrue(torch.equal(regions, full[:, 1:]))
        self.assertEqual(sum(p.requires_grad for p in encoder.parameters()), 0)
        self.assertFalse(encoder.train().training)
        self.assertFalse(regions.requires_grad)

    def test_a_batch_equals_the_images_encoded_one_by_one(self):
        encoder = FrozenCLIPVisionGrid(vision_model=tiny_clip())
        pixels = torch.randn(4, 3, 224, 224)
        single = torch.cat([encoder(pixels[i:i + 1]) for i in range(4)])
        self.assertLess((encoder(pixels) - single).abs().max().item(), 1e-4)

    def test_unsupported_grid_is_rejected(self):
        config = CLIPVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=4,
                                  image_size=224, patch_size=16)
        with self.assertRaisesRegex(ValueError, "grid"):
            FrozenCLIPVisionGrid(vision_model=CLIPVisionModel(config))

    def test_clip_preprocessing_matches_independent_recipe_for_non_square_images(self):
        processor = clip_processor()
        with tempfile.TemporaryDirectory() as directory:
            for name, size in (("wide.png", (680, 356)), ("tall.png", (301, 777)), ("small.png", (100, 150))):
                path = Path(directory) / name
                image = Image.effect_noise(size, 64).convert("RGB")
                image.save(path)
                tensor, digest = load_clip_image(path, processor)
                self.assertEqual(tensor.shape, (3, 224, 224))
                self.assertEqual(digest, pixel_sha256(image))
                self.assertLess((tensor - manual_clip_preprocess(path, processor)).abs().max().item(), 0.05, name)

    def test_clip_normalisation_differs_from_imagenet(self):
        processor = clip_processor()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "white.png"
            Image.new("RGB", (224, 224), (255, 255, 255)).save(path)
            tensor, _ = load_clip_image(path, processor)
        expected = torch.tensor([(1 - m) / s for m, s in zip(MEAN, STD)]).view(3, 1, 1).expand(3, 224, 224)
        self.assertTrue(torch.allclose(tensor, expected, atol=1e-4))


class DecoderFeatureDimTests(unittest.TestCase):
    def test_defaults_are_exactly_model_b(self):
        default = TransformerCaptionDecoder(1716)
        explicit = TransformerCaptionDecoder(1716, feature_dim=FEATURE_DIM, num_regions=NUM_REGIONS)
        self.assertEqual(sum(p.numel() for p in default.parameters()), 15_447_220)
        self.assertEqual(list(default.state_dict()), list(explicit.state_dict()))
        self.assertEqual(default.regions.linear.in_features, 2048)

    def test_clip_variant_changes_only_the_region_projection(self):
        clip = TransformerCaptionDecoder(1716, feature_dim=768)
        default = TransformerCaptionDecoder(1716)
        changed = [k for k, v in clip.state_dict().items() if v.shape != default.state_dict()[k].shape]
        self.assertEqual(changed, ["regions.linear.weight"])
        self.assertEqual(sum(p.numel() for p in default.parameters()) - sum(p.numel() for p in clip.parameters()),
                         512 * (2048 - 768))

    def test_forward_and_greedy_generation_with_clip_regions(self):
        torch.manual_seed(0)
        model = TransformerCaptionDecoder(VOCAB, d_model=32, num_heads=4, num_layers=2, d_ff=64, feature_dim=32)
        regions = torch.randn(3, NUM_REGIONS, 32)
        self.assertEqual(model(regions, torch.tensor([[1, 5, 6]] * 3)).shape, (3, 3, VOCAB))
        tokens = generate_tokens(model, regions, 6)
        self.assertEqual(tokens.shape[0], 3)
        self.assertLessEqual(tokens.shape[1], 6)
        with self.assertRaises(ValueError):
            model(torch.randn(3, NUM_REGIONS, 2048), torch.tensor([[1, 5, 6]] * 3))

    def test_invalid_feature_dim_or_region_count(self):
        with self.assertRaises(ValueError):
            TransformerCaptionDecoder(VOCAB, feature_dim=0)
        with self.assertRaises(ValueError):
            TransformerCaptionDecoder(VOCAB, num_regions=0)

    def test_greedy_decode_stops_at_eos_and_pads_after_it(self):
        class Fixed(torch.nn.Module):
            def forward(self, regions, prefix):
                logits = torch.zeros(prefix.shape[0], prefix.shape[1], VOCAB)
                logits[0, :, 7] = 5.0 if prefix.shape[1] < 3 else -5.0
                logits[0, :, EOS] = 0.0 if prefix.shape[1] < 3 else 5.0
                logits[1, :, 9] = 5.0
                return logits

        tokens = generate_tokens(Fixed(), torch.zeros(2, 1, 1), 5)
        self.assertEqual(tokens[0].tolist(), [7, 7, EOS, PAD, PAD])
        self.assertEqual(tokens[1].tolist(), [9] * 5)


class CachePrefixTests(unittest.TestCase):
    def test_clip_cache_has_its_own_file_and_resnet_default_is_unchanged(self):
        self.assertEqual(feature_path("d", "val").name, "resnet50_7x7_val.pt")
        self.assertEqual(feature_path("d", "val", CLIP_CACHE_PREFIX).name, "clip_vit_b32_7x7_val.pt")
        with tempfile.TemporaryDirectory() as directory:
            payload = {"image_ids": torch.tensor([4, 2]), "features": torch.zeros(2, NUM_REGIONS, 768, dtype=torch.float16),
                       "image_paths": ["a", "b"], "pixel_sha256": ["x", "y"], "meta": {}}
            torch.save(payload, feature_path(directory, "val", CLIP_CACHE_PREFIX))
            store = RegionFeatureStore.load(directory, "val", CLIP_CACHE_PREFIX, feature_dim=768)
            self.assertEqual(store.get_batch([2, 4]).shape, (2, NUM_REGIONS, 768))
            self.assertEqual(store.nearest_id(store.get_batch([4])[0]), (4, 0.0))
            with self.assertRaises(ValueError):
                store.nearest_id(torch.zeros(NUM_REGIONS, FEATURE_DIM))
            with self.assertRaises(FileNotFoundError):
                RegionFeatureStore.load(directory, "val")  # the ResNet cache is a different file


if __name__ == "__main__":
    unittest.main()
