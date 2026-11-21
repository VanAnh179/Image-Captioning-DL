import unittest

from scripts.generate_caption_predictions import resolve_feature_prefix
from src.data.region_features import CACHE_PREFIX


class ResolveFeaturePrefixTests(unittest.TestCase):
    def test_resnet_models_use_default_cache_without_a_feature_dim(self):
        self.assertEqual(resolve_feature_prefix({"d_model": 512}), CACHE_PREFIX)
        self.assertEqual(resolve_feature_prefix({"feature_dim": 2048, "num_regions": 49}), CACHE_PREFIX)

    def test_clip_model_uses_clip_cache(self):
        self.assertEqual(resolve_feature_prefix({"feature_dim": 768, "num_regions": 49}), "clip_vit_b32_7x7")

    def test_explicit_prefix_wins(self):
        self.assertEqual(resolve_feature_prefix({"feature_dim": 768}, "my_cache"), "my_cache")
        self.assertEqual(resolve_feature_prefix({}, "my_cache"), "my_cache")

    def test_unknown_feature_dim_asks_for_a_prefix(self):
        with self.assertRaises(ValueError) as error:
            resolve_feature_prefix({"feature_dim": 1024})
        self.assertIn("--feature-prefix", str(error.exception))


if __name__ == "__main__":
    unittest.main()
