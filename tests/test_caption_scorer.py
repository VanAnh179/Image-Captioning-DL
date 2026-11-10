import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.run_scorer_trial import main as trial_cli
from scripts.score_captions import main as score_cli
from src.evaluation.caption_scorer import (
    ScorerInputError, load_predictions, load_references, score_captions, validate,
)

REFERENCES = {
    1: ["một người phụ nữ đang cõng em bé", "người phụ nữ cõng một đứa trẻ trên lưng",
        "có một em bé trên lưng người phụ nữ"],
    2: ["những chiếc đèn lồng treo trên phố", "có nhiều đèn lồng ở trên phố", "phố có rất nhiều đèn lồng"],
    3: ["hai con chó đang chạy trên bãi cỏ", "bãi cỏ có hai con chó đang chạy", "hai chú chó chạy trên cỏ"],
}


class ScorerTests(unittest.TestCase):
    def test_trial_handles_shared_caption_text_and_propagates_scoring_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for image_id in REFERENCES:
                # Distinct IDs may legitimately have the same first reference.
                captions = REFERENCES[1] if image_id == 2 else REFERENCES[image_id]
                rows.append({"image_id": image_id, "captions": captions, "image_path": f"{image_id}.jpg"})
                (root / f"{image_id}.jpg").touch()
            (root / "val_manifest.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
            argv = ["--processed-dir", str(root), "--image-root", str(root), "--output", str(root / "results")]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(trial_cli(argv), 0)
            with patch("scripts.run_scorer_trial.run_cli", return_value=(2, "", "failed")), \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(trial_cli(argv), 1)
            report = json.loads((root / "results/val_trial_report.json").read_text(encoding="utf-8"))
            self.assertFalse(report["passed"])

    def test_prediction_formats_include_single_line_jsonl(self):
        record = {"image_id": 1, "caption": "một con chó"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            for data in (record, [record], {"1": record["caption"]}):
                path.write_text(json.dumps(data), encoding="utf-8")
                self.assertEqual(load_predictions(path), [(1, record["caption"])])
            path.write_text(json.dumps(record) + "\n" + json.dumps({**record, "image_id": 2}), encoding="utf-8")
            self.assertEqual([i for i, _ in load_predictions(path)], [1, 2])

    def test_reference_loader_keeps_every_caption_and_rejects_duplicate_ids(self):
        record = {"image_id": 1, "captions": ["một con chó"] * 5}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "val_manifest.jsonl"
            path.write_text(json.dumps(record), encoding="utf-8")
            self.assertEqual(len(load_references(path)[1]), 5)
            path.write_text(json.dumps(record) + "\n" + json.dumps(record), encoding="utf-8")
            with self.assertRaisesRegex(ScorerInputError, "ID trùng"):
                load_references(path)
            for bad in ("một câu", [], [""], [123]):
                path.write_text(json.dumps({**record, "captions": bad}), encoding="utf-8")
                with self.assertRaises(ScorerInputError):
                    load_references(path)

    def test_cli_reports_missing_predictions_and_missing_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "val_manifest.jsonl"
            rows = [{"image_id": i, "captions": refs, "image_path": f"{i}.jpg"} for i, refs in REFERENCES.items()]
            manifest.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
            path = root / "predictions.json"
            for records, message in (([{"image_id": 1, "caption": "a"}], "Thiếu dự đoán"),
                                     ([{"image_id": i, "caption": refs[0]} for i, refs in REFERENCES.items()], "Thiếu file ảnh")):
                path.write_text(json.dumps(records), encoding="utf-8")
                error = io.StringIO()
                with contextlib.redirect_stderr(error):
                    code = score_cli(["--processed-dir", str(root), "--image-root", str(root), "--predictions", str(path)])
                self.assertEqual(code, 2)
                self.assertIn(message, error.getvalue())

    def test_correct_ids_score_far_above_swapped_ids(self):
        correct = {i: refs[0] for i, refs in REFERENCES.items()}
        swapped = {1: correct[2], 2: correct[3], 3: correct[1]}
        good = score_captions(list(correct.items()), REFERENCES)["scores"]
        bad = score_captions(list(swapped.items()), REFERENCES)["scores"]
        self.assertEqual(good["BLEU-4"], 100.0)
        for metric in good:
            self.assertGreater(good[metric], bad[metric])
        result = score_captions(list(correct.items()), REFERENCES)
        self.assertEqual(result["num_references"], 9)
        self.assertIn("CIDEr-D 0-1000", result["scale"])

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
