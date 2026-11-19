"""Compare model B (ResNet-50 regions) with model C (frozen CLIP regions) on identical validation data.

Writes a JSON table with scores, training cost and examples, and optionally an image with six examples.
"""

import argparse
import json
import math
import textwrap
from pathlib import Path

import numpy as np
from pycocoevalcap.cider.cider import Cider

from src.data.region_features import feature_path, read_manifest
from src.data.vietnamese_caption_batch import normalize_caption
from src.evaluation.caption_scorer import score_captions
from src.training.caption_trainer import build_model


def load_run(directory):
    directory = Path(directory)
    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    history = json.loads((directory / "history.json").read_text(encoding="utf-8"))
    records = json.loads((directory / "val_predictions.json").read_text(encoding="utf-8"))
    return metrics, history, {row["image_id"]: row["caption"] for row in records}


def run_cost(directory, metrics, history, prefix, features_dir):
    directory = Path(directory)
    config = metrics["config"]
    parameters = sum(p.numel() for p in build_model("B", config).parameters())
    train_seconds = [row["train"]["seconds"] for row in history]
    cache_bytes = sum(feature_path(features_dir, split, prefix).stat().st_size for split in ("train", "val", "test"))
    return {"parameters": parameters, "epochs_run": len(history), "best_epoch": metrics["best_epoch"],
            "train_seconds_total": round(sum(train_seconds), 1),
            "train_seconds_per_epoch": round(sum(train_seconds) / len(train_seconds), 1),
            "peak_vram_allocated_mib_train": round(max(row["train"]["peak_allocated_mib"] for row in history), 1),
            "best_checkpoint_mib": round((directory / "best.pt").stat().st_size / 2**20, 1),
            "feature_cache_mib_all_splits": round(cache_bytes / 2**20, 1), "device": metrics["device"]}


def per_image_cider(predictions, references):
    gts = {i: [normalize_caption(c) for c in refs] for i, refs in references.items()}
    _, values = Cider().compute_score(gts, {i: [normalize_caption(predictions[i])] for i in gts})
    return {i: float(v) * 100 for i, v in zip(gts, values)}


def bootstrap_mean(differences, resamples=2000, seed=0):
    values = np.asarray(differences, dtype=float)
    draws = np.random.default_rng(seed).integers(0, len(values), size=(resamples, len(values)))
    low, high = np.percentile(values[draws].mean(axis=1), [2.5, 97.5])
    return {"mean": float(values.mean()), "ci95_low": float(low), "ci95_high": float(high), "resamples": resamples}


def draw_examples(examples, image_root, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image, ImageOps

    rows = math.ceil(len(examples) / 2)
    fig = plt.figure(figsize=(15, rows * 4.8), layout="constrained")
    grid = fig.add_gridspec(rows, 2)
    for position, example in enumerate(examples):
        cell = grid[position // 2, position % 2].subgridspec(2, 1, height_ratios=[3, 1.3])
        image_ax, text_ax = fig.add_subplot(cell[0]), fig.add_subplot(cell[1])
        with Image.open(Path(image_root) / example["image_path"]) as source:
            image_ax.imshow(ImageOps.exif_transpose(source).convert("RGB"))
        image_ax.set_title(f"ID {example['image_id']} | CIDEr C - B = {example['delta_c_minus_b']:+.2f}")
        image_ax.axis("off")
        text = "\n\n".join(textwrap.fill(label + value, width=78) for label, value in (
            ("B: ", example["caption_b"]), ("C: ", example["caption_c"]), ("Tham chiếu: ", example["references"][0])))
        text_ax.text(0, 1, text, va="top", fontsize=10)
        text_ax.axis("off")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b-dir", type=Path, required=True)
    parser.add_argument("--c-dir", type=Path, required=True)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed/ktvic"))
    parser.add_argument("--features", type=Path, default=Path("artifacts/features/ktvic"))
    parser.add_argument("--image-root", type=Path, default=Path("data/raw/ktvic"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path)
    args = parser.parse_args()

    b_metrics, b_history, b_captions = load_run(args.b_dir)
    c_metrics, c_history, c_captions = load_run(args.c_dir)
    if b_metrics["settings"] != c_metrics["settings"]:
        raise ValueError("B and C were trained with different settings")
    if {k: v for k, v in c_metrics["config"].items() if k not in ("feature_dim", "num_regions")} != \
            {k: v for k, v in b_metrics["config"].items() if k not in ("feature_dim", "num_regions")}:
        raise ValueError("B and C differ in more than the feature shape")
    rows = read_manifest(args.processed_dir / "val_manifest.jsonl")
    references = {row["image_id"]: row["captions"] for row in rows}
    paths = {row["image_id"]: row["image_path"] for row in rows}
    scores = {}
    for name, captions, metrics in (("B", b_captions, b_metrics), ("C", c_captions, c_metrics)):
        scored = score_captions([(i, captions[i]) for i in references], references, allow_empty=True)
        if scored["scores"] != metrics["scores"]["scores"]:
            raise ValueError(f"{name}: recomputed scores differ from the selected checkpoint metrics")
        scores[name] = scored["scores"]
    cider_b, cider_c = per_image_cider(b_captions, references), per_image_cider(c_captions, references)
    delta = {i: cider_c[i] - cider_b[i] for i in references}
    changed = [i for i in references if b_captions[i] != c_captions[i]]
    c_wins = sorted((i for i in changed if delta[i] > 0), key=lambda i: (-delta[i], i))[:3]
    b_wins = sorted((i for i in changed if delta[i] < 0), key=lambda i: (delta[i], i))[:3]
    examples = [{"image_id": i, "image_path": paths[i], "caption_b": b_captions[i], "caption_c": c_captions[i],
                 "references": references[i], "cider_b": cider_b[i], "cider_c": cider_c[i], "delta_c_minus_b": delta[i]}
                for i in c_wins + b_wins]
    report = {"images": len(references), "scores": scores,
              "c_minus_b": {name: round(scores["C"][name] - scores["B"][name], 2) for name in scores["B"]},
              "changed_captions": len(changed),
              "images_c_better": sum(delta[i] > 0 for i in references), "images_b_better": sum(delta[i] < 0 for i in references),
              "images_equal": sum(delta[i] == 0 for i in references),
              "mean_per_image_cider_difference_c_minus_b": bootstrap_mean([delta[i] for i in references]),
              "cost": {"B": run_cost(args.b_dir, b_metrics, b_history, "resnet50_7x7", args.features),
                       "C": run_cost(args.c_dir, c_metrics, c_history, "clip_vit_b32_7x7", args.features)},
              "examples": examples,
              "selection": "three largest per-image CIDEr gains for each model among changed captions; illustrative, not representative"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.figure and examples:
        draw_examples(examples, args.image_root, args.figure)
    print(json.dumps({k: v for k, v in report.items() if k != "examples"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
