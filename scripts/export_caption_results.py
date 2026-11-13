"""Export the selected checkpoint's validation metrics, captions and loss curve."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


def export_results(run_dir):
    run_dir = Path(run_dir)
    best = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    last = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=True)
    if best["validation_predictions"] is None:
        raise ValueError("Selected checkpoint has no caption evaluation; train with selection-metric cider")
    predictions = best["validation_predictions"]
    (run_dir / "val_predictions.json").write_text(json.dumps(predictions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    metrics = {"best_epoch": best["epoch"], "last_epoch": last["epoch"], "scores": best["validation_scores"],
               "settings": best["settings"], "config": best["config"], "device": best["device"],
               "torch": best["torch"], "data_signature": best["data_signature"],
               "train_seconds": sum(row["train"]["seconds"] for row in last["history"]),
               "validation_loss_seconds": sum(row["validation"]["seconds"] for row in last["history"]),
               "validation_caption_seconds": sum(row["caption_seconds"] for row in last["history"])}
    (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    history = last["history"]
    fig, ax = plt.subplots(figsize=(7, 4))
    for key, label in (("train", "Train"), ("validation", "Validation")):
        ax.plot([row["epoch"] for row in history], [row[key]["loss"] for row in history], label=label)
    ax.set(xlabel="Epoch", ylabel="Cross-entropy per valid token")
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "loss_curve.png", dpi=160)
    plt.close(fig)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    export_results(parser.parse_args().run_dir)
