"""Trial run of the caption scorer: correct-ID vs shuffled-ID predictions, plus the error messages."""

import argparse
import contextlib
import io
import json
import random
import shutil
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("runs/scorer_trial"))
    args = parser.parse_args()

    processed = resolve_processed_dir()
    image_root = resolve_image_root()
    references = load_references(processed / f"{args.split}_manifest.jsonl")
    ids = sorted(references)
    # Prediction = the first human reference of the image. Because it is one of the five references, the
    # correct-ID case is an upper bound that checks the scorer, not an estimate of model quality.
    own = {i: references[i][0] for i in ids}
    order = ids[:]
    random.Random(args.seed).shuffle(order)
    swapped = {order[j]: own[order[(j + 1) % len(order)]] for j in range(len(order))}  # nobody keeps its own caption
    assert all(swapped[i] != own[i] for i in ids)  # a derangement: nobody keeps its own caption

    report = {"split": args.split, "num_images": len(ids), "seed": args.seed, "cases": {}}
    for name, mapping in (("dung_ID", own), ("tráo_ID", swapped)):
        path = args.output / f"{args.split}_predictions_{name}.json"
        write_predictions(path, mapping.items())
        code, out, err = run_cli("--predictions", path, "--split", args.split, "--output",
                                 args.output / f"{args.split}_scores_{name}.json")
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
        code, out, err = run_cli("--predictions", path, "--split", args.split)
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
        code, out, err = run_cli("--predictions", path, "--split", args.split, "--image-root", root)
        report["cases"]["lỗi: thiếu file ảnh"] = {"exit_code": code, "output": out or err}
        print(f"=== lỗi: thiếu file ảnh (exit {code}) ===\n{out or err}\n")

    (args.output / f"{args.split}_trial_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
