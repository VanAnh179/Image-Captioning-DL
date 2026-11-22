import unittest

from scripts.build_test_score_table import check_manifest_order, check_model_variant, markdown_table, vi


class TestScoreTableTests(unittest.TestCase):
    def test_vietnamese_number_style(self):
        self.assertEqual(vi(72.4), "72,40")
        self.assertEqual(vi(15477236, 0), "15.477.236")
        self.assertEqual(vi(1234.56, 1), "1.234,6")

    def test_captions_must_follow_test_manifest_order(self):
        check_manifest_order([(1, "a"), (2, "b")], [1, 2], "A")
        for pairs in ([(2, "b"), (1, "a")], [(1, "a")], [(1, "a"), (2, "b"), (3, "c")]):
            with self.assertRaises(ValueError):
                check_manifest_order(pairs, [1, 2], "A")

    def test_there_are_distinct_b_and_c_feature_caches(self):
        check_model_variant({"model_name": "B", "feature_prefix": "clip_vit_b32_7x7"}, "C")
        with self.assertRaisesRegex(ValueError, "C: expected"):
            check_model_variant({"model_name": "B", "feature_prefix": "resnet50_7x7"}, "C")

    def test_markdown_row_has_scores_parameters_and_times(self):
        scores = {"BLEU-1": 60.0, "BLEU-2": 45.5, "BLEU-3": 34.25, "BLEU-4": 26.0, "CIDEr": 70.5}
        info = {"val_scores": {"CIDEr": 72.42}, "parameters": 15447220, "selected_epoch": 6,
                "train_seconds_to_selected_epoch": 212.3, "generation_seconds": 8.04}
        text = markdown_table([{"model": "B", "info": info, "test_scores": scores}], 558, 2790)
        self.assertIn("558 ảnh, 2.790 câu tham chiếu", text)
        row = text.splitlines()[-1]
        for expected in ("| B |", "ResNet-50", "**70,50**", "72,42", "15.447.220", "| 6 |", "212,3", "8,0"):
            self.assertIn(expected, row)


if __name__ == "__main__":
    unittest.main()
