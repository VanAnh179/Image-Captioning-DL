"""Test-split table for the locked models A, B, A0 and C: scores recomputed from saved caption files.

Recreate the table from the saved files (no checkpoints needed):
  python -m scripts.build_test_score_table
Regenerate the captions from the locked checkpoints first, then rebuild the table:
  python -m scripts.build_test_score_table --generate --checkpoint A=<best.pt> --checkpoint B=<best.pt> \
      --checkpoint A0=<best.pt> --checkpoint C=<best.pt>
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch

from scripts.generate_caption_predictions import generate_predictions
from src.data.region_features import read_manifest
from src.evaluation.caption_scorer import SCORE_SCALE, library_versions, load_predictions, score_captions
from src.training.caption_trainer import load_checkpoint

MODELS = ("A", "B", "A0", "C")
ENCODERS = {"A": "ResNet-50", "B": "ResNet-50", "A0": "ResNet-50", "C": "CLIP ViT-B/32"}
METRICS = ("BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "CIDEr")


def sha256_of(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def training_info(checkpoint_path):
    """What the checkpoint recorded about its own training run on validation data."""
    checkpoint = load_checkpoint(checkpoint_path)
    history = checkpoint["history"]
    return {"checkpoint_sha256": sha256_of(checkpoint_path), "model_name": checkpoint["model_name"],
            "selected_epoch": checkpoint["epoch"],
            "val_scores": checkpoint["validation_scores"]["scores"], "seed": checkpoint["settings"]["seed"],
            "train_seconds_to_selected_epoch": round(sum(row["train"]["seconds"] for row in history), 1),
            "train_device": checkpoint["device"], "train_torch": checkpoint["torch"]}


def check_manifest_order(pairs, test_ids, name):
    ids = [image_id for image_id, _ in pairs]
    if ids != test_ids:
        raise ValueError(f"{name}: captions must cover every test ID once, in test manifest order")


def vi(value, digits=2):
    """Vietnamese number style: comma for decimals, dot for thousands."""
    text = f"{value:,.{digits}f}"
    return text.replace(",", "\0").replace(".", ",").replace("\0", ".")


def markdown_table(rows, images, references):
    lines = [f"Tập test: {images} ảnh, {vi(references, 0)} câu tham chiếu (5 câu mỗi ảnh). Điểm nhân 100 "
             "(BLEU-1..4 và CIDEr-D, `pycocoevalcap`).", "",
             "| Mô hình | Encoder | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | CIDEr | CIDEr val | Tham số học được | "
             "Epoch chọn | Train tới epoch chọn (giây) | Sinh 558 caption test (giây) |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        info, scores = row["info"], row["test_scores"]
        lines.append(
            f"| {row['model']} | {ENCODERS[row['model']]} | " + " | ".join(vi(scores[m]) for m in METRICS[:4])
            + f" | **{vi(scores['CIDEr'])}** | {vi(info['val_scores']['CIDEr'])} | {vi(info['parameters'], 0)} | "
            f"{info['selected_epoch']} | {vi(info['train_seconds_to_selected_epoch'], 1)} | "
            f"{vi(info['generation_seconds'], 1)} |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/test_results"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed/ktvic"))
    parser.add_argument("--features", type=Path, default=Path("artifacts/features/ktvic"))
    parser.add_argument("--generate", action="store_true", help="Generate the test captions from the checkpoints first")
    parser.add_argument("--checkpoint", action="append", default=[], metavar="MODEL=PATH")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    manifest = args.processed_dir / "test_manifest.jsonl"
    rows = read_manifest(manifest)
    test_ids = [row["image_id"] for row in rows]
    references = {row["image_id"]: row["captions"] for row in rows}
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    if args.generate:
        paths = dict(item.split("=", 1) for item in args.checkpoint)
        if sorted(paths) != sorted(MODELS):
            parser.error("--generate needs --checkpoint for each of " + ", ".join(MODELS))
        for name in MODELS:
            records, generation = generate_predictions(Path(paths[name]), "test", args.device, args.processed_dir,
                                                       args.features)
            info = {**training_info(paths[name]), **generation}
            (out / f"{name}_test_predictions.json").write_text(
                json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            (out / f"{name}_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"{name}: {len(records)} captions, {generation['generation_seconds']} s", flush=True)

    table = []
    for name in MODELS:
        prediction_file = out / f"{name}_test_predictions.json"
        info = json.loads((out / f"{name}_info.json").read_text(encoding="utf-8"))
        pairs = load_predictions(prediction_file)
        check_manifest_order(pairs, test_ids, name)
        scored = score_captions(pairs, references)
        table.append({"model": name, "images": len(pairs), "predictions_sha256": sha256_of(prediction_file),
                      "test_scores": scored["scores"], "info": info})
    result = {"split": "test", "images": len(test_ids), "references": sum(len(v) for v in references.values()),
              "scale": SCORE_SCALE, "test_manifest_sha256": sha256_of(manifest), "git_commit": git_commit(),
              "libraries": library_versions(), "python": sys.version.split()[0], "models": table}
    (out / "test_table.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown = markdown_table(table, result["images"], result["references"])
    (out / "test_table.md").write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
