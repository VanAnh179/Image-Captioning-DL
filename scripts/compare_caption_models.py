"""Compare A with a freshly trained uniform-attention A0 on identical validation data."""

import argparse
import hashlib
import json
from pathlib import Path

from pycocoevalcap.cider.cider import Cider

from src.data.region_features import read_manifest
from src.data.vietnamese_caption_batch import normalize_caption
from src.evaluation.caption_scorer import score_captions
from src.training.caption_trainer import load_checkpoint


def compare_models(reference_dir, ablation_dir, processed_dir):
    reference_dir, ablation_dir = Path(reference_dir), Path(ablation_dir)
    reference = json.loads((reference_dir / "metrics.json").read_text(encoding="utf-8"))
    ablation = json.loads((ablation_dir / "metrics.json").read_text(encoding="utf-8"))
    base_config = {k: v for k, v in ablation["config"].items() if k != "attention_mode"}
    if base_config != reference["config"] or ablation["config"].get("attention_mode") != "uniform":
        raise ValueError("A0 must differ from A only in attention_mode=uniform")
    for key in ("settings", "data_signature"):
        if reference[key] != ablation[key]:
            raise ValueError(f"A and A0 differ in {key}")
    manifest = Path(processed_dir) / "val_manifest.jsonl"
    rows = read_manifest(manifest)
    references = {row["image_id"]: row["captions"] for row in rows}
    predictions = []
    metrics = []
    per_image = []
    gts = {i: [normalize_caption(c) for c in refs] for i, refs in references.items()}
    for directory, saved in ((reference_dir, reference), (ablation_dir, ablation)):
        records = json.loads((directory / "val_predictions.json").read_text(encoding="utf-8"))
        pairs = [(row["image_id"], row["caption"]) for row in records]
        scored = score_captions(pairs, references, allow_empty=True)
        if scored["scores"] != saved["scores"]["scores"]:
            raise ValueError("Recomputed caption scores differ from the selected checkpoint metrics")
        captions = dict(pairs)
        predictions.append(captions)
        metrics.append(scored)
        _, values = Cider().compute_score(gts, {i: [normalize_caption(captions[i])] for i in gts})
        per_image.append({i: float(score) * 100 for i, score in zip(gts, values)})
    changed = [i for i in gts if predictions[0][i] != predictions[1][i]]
    differences = {i: per_image[0][i] - per_image[1][i] for i in changed}
    a_wins = sorted((i for i in changed if differences[i] > 0), key=lambda i: (-differences[i], i))[:3]
    a0_wins = sorted((i for i in changed if differences[i] < 0), key=lambda i: (differences[i], i))[:3]
    chosen = list(dict.fromkeys(a_wins + a0_wins))
    if len(chosen) < 6:
        remaining = sorted((i for i in changed if i not in chosen), key=lambda i: (-abs(differences[i]), i))
        chosen += remaining[:6 - len(chosen)]
    paths = {row["image_id"]: row["image_path"] for row in rows}
    examples = [{"image_id": i, "image_path": paths[i], "caption_a": predictions[0][i],
                 "caption_a0": predictions[1][i], "references": references[i],
                 "cider_a": per_image[0][i], "cider_a0": per_image[1][i], "delta_a_minus_a0": differences[i]}
                for i in chosen]
    best = load_checkpoint(ablation_dir / "best.pt")
    report = {"reference_best_epoch": reference["best_epoch"], "ablation_best_epoch": ablation["best_epoch"],
              "a_scores": metrics[0], "a0_scores": metrics[1],
              "delta_a_minus_a0": {name: round(metrics[0]["scores"][name] - metrics[1]["scores"][name], 2) for name in metrics[0]["scores"]},
              "changed_captions": len(changed), "images": len(rows), "examples": examples,
              "selection": "up to three largest per-image CIDEr wins for each model; illustrative selection",
              "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
              "checkpoint_model": best["model_name"], "attention_mode": best["config"]["attention_mode"]}
    (ablation_dir / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "examples"}, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--ablation-dir", type=Path, required=True)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed/ktvic"))
    args = parser.parse_args()
    compare_models(args.reference_dir, args.ablation_dir, args.processed_dir)
