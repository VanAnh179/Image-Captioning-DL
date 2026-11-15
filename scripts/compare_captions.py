"""List validation images that model A and model B describe differently, with per-image CIDEr for each."""

import argparse
import json
import sys
from pathlib import Path

from pycocoevalcap.cider.cider import Cider

from src.data.paths import resolve_processed_dir
from src.data.region_features import read_manifest
from src.data.vietnamese_caption_batch import normalize_caption
from src.evaluation.caption_scorer import load_predictions, load_references


def overlap(a, b):
    a, b = set(a.split()), set(b.split())
    return len(a & b) / len(a | b) if a | b else 1.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", type=Path, required=True, help="val_predictions.json of model A")
    parser.add_argument("--b", type=Path, required=True, help="val_predictions.json of model B")
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--each", type=int, default=3, help="IDs to list where A is clearly better, and where B is")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--processed-dir", type=Path)
    args = parser.parse_args()

    processed = args.processed_dir or resolve_processed_dir()
    manifest = processed / f"{args.split}_manifest.jsonl"
    references = load_references(manifest)
    paths = {row["image_id"]: row["image_path"] for row in read_manifest(manifest)}
    pred_a, pred_b = dict(load_predictions(args.a)), dict(load_predictions(args.b))
    ids = sorted(references)
    if set(pred_a) != set(ids) or set(pred_b) != set(ids):
        raise SystemExit("Cả hai file phải có đúng một caption cho mọi ID của tập")

    gts = {i: [normalize_caption(r) for r in references[i]] for i in ids}
    per_image = {}
    for name, pred in (("A", pred_a), ("B", pred_b)):
        _, scores = Cider().compute_score(gts, {i: [normalize_caption(pred[i])] for i in ids})
        per_image[name] = dict(zip(ids, (float(s) * 100 for s in scores)))

    rows = []
    for i in ids:
        a, b = normalize_caption(pred_a[i]), normalize_caption(pred_b[i])
        rows.append({"image_id": i, "image_path": paths[i], "caption_A": a, "caption_B": b,
                     "cider_A": round(per_image["A"][i], 1), "cider_B": round(per_image["B"][i], 1),
                     "word_overlap": round(overlap(a, b), 2), "references": references[i]})
    different = [r for r in rows if r["caption_A"] != r["caption_B"]]
    by_gap = sorted(different, key=lambda r: r["cider_B"] - r["cider_A"])
    a_better, b_better = by_gap[:args.each], by_gap[::-1][:args.each]
    summary = {
        "split": args.split, "images": len(ids), "identical_captions": len(rows) - len(different),
        "different_captions": len(different),
        "mean_word_overlap": round(sum(r["word_overlap"] for r in rows) / len(rows), 3),
        "images_A_better": sum(r["cider_A"] > r["cider_B"] for r in rows),
        "images_B_better": sum(r["cider_B"] > r["cider_A"] for r in rows),
        "images_equal": sum(r["cider_A"] == r["cider_B"] for r in rows),
        "a_file": str(args.a), "b_file": str(args.b),
    }
    result = {"summary": summary, "A_clearly_better": a_better, "B_clearly_better": b_better}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    for title, group in (("A tốt hơn rõ", a_better), ("B tốt hơn rõ", b_better)):
        print(f"\n== {title}")
        for r in group:
            print(f"ID {r['image_id']}  CIDEr A {r['cider_A']} | B {r['cider_B']}\n  A: {r['caption_A']}\n  B: {r['caption_B']}\n  tham chiếu: {r['references'][0]}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
