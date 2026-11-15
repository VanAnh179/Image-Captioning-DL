import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.compare_captions import main

REFERENCES = {
    1: ["một người phụ nữ đang cõng em bé", "người phụ nữ cõng một đứa trẻ trên lưng"],
    2: ["những chiếc đèn lồng treo trên phố", "có nhiều đèn lồng ở trên phố"],
    3: ["hai con chó đang chạy trên bãi cỏ", "bãi cỏ có hai con chó đang chạy"],
}
WRONG = "xin chào các bạn"


def run(directory, pred_a, pred_b, each=1):
    directory = Path(directory)
    manifest = directory / "val_manifest.jsonl"
    manifest.write_text("\n".join(json.dumps({"image_id": i, "image_path": f"{i}.jpg", "captions": refs},
                                             ensure_ascii=False) for i, refs in REFERENCES.items()), encoding="utf-8")
    files = {}
    for name, predictions in (("a", pred_a), ("b", pred_b)):
        files[name] = directory / f"{name}.json"
        files[name].write_text(json.dumps([{"image_id": i, "caption": c} for i, c in predictions.items()],
                                          ensure_ascii=False), encoding="utf-8")
    output = directory / "out.json"
    argv = ["compare_captions", "--a", str(files["a"]), "--b", str(files["b"]), "--output", str(output),
            "--processed-dir", str(directory), "--each", str(each)]
    with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
        main()
    return json.loads(output.read_text(encoding="utf-8"))


class CompareCaptionsTests(unittest.TestCase):
    def test_counts_and_lists_follow_the_per_image_scores(self):
        a = {1: REFERENCES[1][0], 2: REFERENCES[2][0], 3: WRONG}
        b = {1: WRONG, 2: REFERENCES[2][0], 3: REFERENCES[3][0]}
        with tempfile.TemporaryDirectory() as directory:
            result = run(directory, a, b)
        summary = result["summary"]
        self.assertEqual((summary["identical_captions"], summary["different_captions"]), (1, 2))
        self.assertEqual((summary["images_A_better"], summary["images_B_better"], summary["images_equal"]), (1, 1, 1))
        self.assertEqual([row["image_id"] for row in result["A_clearly_better"]], [1])
        self.assertEqual([row["image_id"] for row in result["B_clearly_better"]], [3])
        self.assertGreater(result["A_clearly_better"][0]["cider_A"], result["A_clearly_better"][0]["cider_B"])

    def test_both_files_must_cover_every_image(self):
        a = {1: REFERENCES[1][0], 2: REFERENCES[2][0], 3: REFERENCES[3][0]}
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(SystemExit):
            run(directory, a, {1: REFERENCES[1][0], 2: REFERENCES[2][0]})


if __name__ == "__main__":
    unittest.main()
