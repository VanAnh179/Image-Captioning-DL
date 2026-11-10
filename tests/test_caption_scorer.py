import tempfile
import unittest
from pathlib import Path

from src.evaluation.caption_scorer import ScorerInputError, score_captions, validate

REFERENCES = {
    1: ["một người phụ nữ đang cõng em bé", "người phụ nữ cõng một đứa trẻ trên lưng",
        "có một em bé trên lưng người phụ nữ"],
    2: ["những chiếc đèn lồng treo trên phố", "có nhiều đèn lồng ở trên phố", "phố có rất nhiều đèn lồng"],
    3: ["hai con chó đang chạy trên bãi cỏ", "bãi cỏ có hai con chó đang chạy", "hai chú chó chạy trên cỏ"],
}


class ScorerTests(unittest.TestCase):
    def test_correct_ids_score_far_above_swapped_ids(self):
        correct = {i: refs[0] for i, refs in REFERENCES.items()}
        swapped = {1: correct[2], 2: correct[3], 3: correct[1]}
        good = score_captions(list(correct.items()), REFERENCES)["scores"]
        bad = score_captions(list(swapped.items()), REFERENCES)["scores"]
        self.assertEqual(good["BLEU-4"], 100.0)
        for metric in good:
            self.assertGreater(good[metric], bad[metric])

    def test_text_is_normalized_like_the_vocabulary(self):
        loud = {i: refs[0].upper() + " !" for i, refs in REFERENCES.items()}
        self.assertEqual(score_captions(list(loud.items()), REFERENCES)["scores"]["BLEU-1"], 100.0)

    def test_missing_unknown_repeated_and_empty_are_all_reported(self):
        with self.assertRaises(ScorerInputError) as caught:
            validate([(1, "câu"), (1, "câu khác"), (99, "lạ"), (2, "  ,. ")], REFERENCES)
        message = str(caught.exception)
        for fragment in ("nhiều hơn một dự đoán", "không thuộc tập tham chiếu", "Thiếu dự đoán cho 1/3", "rỗng"):
            self.assertIn(fragment, message)

    def test_missing_image_file_is_named(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "a.jpg").write_bytes(b"")
            paths = {1: "a.jpg", 2: "b.jpg", 3: "c.jpg"}
            predictions = [(i, refs[0]) for i, refs in REFERENCES.items()]
            with self.assertRaisesRegex(ScorerInputError, r"Thiếu file ảnh.*2, 3"):
                validate(predictions, REFERENCES, image_paths=paths, image_root=directory)


if __name__ == "__main__":
    unittest.main()
