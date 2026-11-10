"""Trial run of the caption scorer: correct-ID vs shuffled-ID predictions, plus the error messages."""

import argparse
import contextlib
import io
import json
import random
import shutil
import sys
import tempfile
from pathlib import Path

from scripts.score_captions import main as score_cli
from src.data.paths import resolve_image_root, resolve_processed_dir
from src.evaluation.caption_scorer import load_references


def write_predictions(path, mapping):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([{"image_id": i, "caption": c} for i, c in mapping], ensure_ascii=False, indent=1),
                    encoding="utf-8")


def run_cli(*argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = score_cli([str(a) for a in argv])
    return code, out.getvalue().strip(), err.getvalue().strip()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("runs/scorer_trial"))
    parser.add_argument("--processed-dir", type=Path)
    parser.add_argument("--image-root", type=Path)
    args = parser.parse_args(argv)

    processed = args.processed_dir or resolve_processed_dir()
    image_root = args.image_root or resolve_image_root()
    references = load_references(processed / f"{args.split}_manifest.jsonl")
    ids = sorted(references)
    if len(ids) < 2:
        parser.error("Cần ít nhất hai ảnh để thử tráo ID và tính CIDEr.")
    # Prediction = the first human reference of the image. Because it is one of the five references, the
    # correct-ID case is a scorer sanity check. It does not measure model quality.
    own = {i: references[i][0] for i in ids}
    order = ids[:]
    random.Random(args.seed).shuffle(order)
    swapped = {order[j]: own[order[(j + 1) % len(order)]] for j in range(len(order))}  # nobody keeps its own caption
    # IDs are deranged even when two distinct images have identical caption text.

    report = {"split": args.split, "num_images": len(ids), "seed": args.seed, "cases": {}}
    failures, scores = [], {}
    common = ("--processed-dir", processed, "--image-root", image_root)
    for name, mapping in (("dung_ID", own), ("tráo_ID", swapped)):
        path = args.output / f"{args.split}_predictions_{name}.json"
        write_predictions(path, mapping.items())
        result_path = args.output / f"{args.split}_scores_{name}.json"
        code, out, err = run_cli(*common, "--predictions", path, "--split", args.split, "--output", result_path)
        if code != 0:
            failures.append(f"{name}: kỳ vọng exit 0, nhận {code}")
        else:
            result = json.loads(result_path.read_text(encoding="utf-8"))
            scores[name] = result["scores"]
            if result["num_references"] != sum(len(r) for r in references.values()):
                failures.append(f"{name}: bộ chấm không dùng đủ câu tham chiếu")
        report["cases"][name] = {"exit_code": code, "output": out or err}
        print(f"=== {name} (exit {code}) ===\n{out or err}\n")

    # Error cases, each from an otherwise valid prediction file.
    valid = list(own.items())
    broken = {
        "thiếu 5 ảnh": valid[:-5],
    }
    for name, mapping in broken.items():
        path = args.output / f"loi_{name.replace(' ', '_')}.json"
        write_predictions(path, mapping)
        code, out, err = run_cli(*common, "--predictions", path, "--split", args.split)
        if code != 2 or "Thiếu dự đoán" not in err:
            failures.append(f"{name}: thiếu thông báo lỗi hoặc exit code 2")
        report["cases"]["lỗi: " + name] = {"exit_code": code, "output": out or err}
        print(f"=== lỗi: {name} (exit {code}) ===\n{out or err}\n")

    with tempfile.TemporaryDirectory() as folder:  # image folder missing three files
        root = Path(folder)
        rows = [json.loads(line) for line in
                (processed / f"{args.split}_manifest.jsonl").read_text(encoding="utf-8").splitlines()]
        for row in rows[3:]:
            target = root / row["image_path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_root / row["image_path"], target)
        path = args.output / "dung_ID_cho_thieu_anh.json"
        write_predictions(path, valid)
        code, out, err = run_cli("--processed-dir", processed, "--predictions", path,
                                 "--split", args.split, "--image-root", root)
        if code != 2 or "Thiếu file ảnh" not in err:
            failures.append("Thiếu file ảnh: thiếu thông báo lỗi hoặc exit code 2")
        report["cases"]["lỗi: thiếu file ảnh"] = {"exit_code": code, "output": out or err}
        print(f"=== lỗi: thiếu file ảnh (exit {code}) ===\n{out or err}\n")

    if len(scores) == 2:
        for metric in scores["dung_ID"]:
            if scores["dung_ID"][metric] <= scores["tráo_ID"][metric]:
                failures.append(f"{metric}: điểm đúng ID phải cao hơn điểm tráo ID")
    report.update(passed=not failures, failures=failures)
    (args.output / f"{args.split}_trial_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    sys.exit(main())
