"""Recall@1/5/10 of TF-IDF image search over A's and B's generated validation captions.

All validation images form the candidate pool. Each of their reference captions is a query and the
image it describes is the answer. One train-fitted index per model is built from that model's captions.
"""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from src.data.region_features import read_manifest
from src.evaluation.caption_scorer import load_predictions
from src.retrieval.caption_search import CaptionSearchIndex
from src.retrieval.recall_eval import CUTOFFS, paired_bootstrap, rank_queries, recall_table


def build_and_evaluate(name, predictions_path, train_rows, val_rows, index_path, rank_path, queries, metadata):
    metadata = {**metadata, "model": name, "predictions_sha256": hashlib.sha256(predictions_path.read_bytes()).hexdigest()}
    index = CaptionSearchIndex.build(train_rows, load_predictions(predictions_path), val_rows, metadata)
    index.save(index_path)
    rows = rank_queries(index, [(image_id, text) for image_id, _, text in queries])
    # Independent check: the public search function must give the same rank for every usable query.
    for (image_id, _, text), row in zip(queries, rows):
        if row["zero_query"]:
            continue
        ids = [hit["image_id"] for hit in index.search(text, k=len(index.records))]
        if ids.index(image_id) + 1 != row["rank"]:
            raise AssertionError(f"{name}: search() and rank_queries disagree for image {image_id}: {text!r}")
    with rank_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["query_number", "image_id", "reference_number", "query", "rank", "rank_best", "rank_worst",
                         "tied_with_answer", "correct_score", "top1_image_id", "zero_query"])
        for number, ((image_id, reference, text), row) in enumerate(zip(queries, rows), 1):
            writer.writerow([number, image_id, reference, text, row["rank"], row["rank_best"], row["rank_worst"],
                             row["tied_with_answer"], f"{row['correct_score']:.6f}", row["top1_image_id"],
                             int(row["zero_query"])])
    table = recall_table(rows, len(index.records))
    table["distinct_generated_captions"] = len({record["caption"] for record in index.records})
    table["predictions_sha256"] = metadata["predictions_sha256"]
    table["index_vocabulary_size"] = len(index.vocabulary)
    return table, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-a", type=Path, required=True)
    parser.add_argument("--predictions-b", type=Path, required=True)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed/ktvic"))
    parser.add_argument("--index-dir", type=Path, default=Path("artifacts/retrieval"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/retrieval_recall"))
    args = parser.parse_args()
    train_path, val_path = args.processed_dir / "train_manifest.jsonl", args.processed_dir / "val_manifest.jsonl"
    train_rows, val_rows = read_manifest(train_path), read_manifest(val_path)
    queries = [(row["image_id"], number, caption)
               for row in val_rows for number, caption in enumerate(row["captions"], 1)]
    metadata = {"train_manifest_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
                "val_manifest_sha256": hashlib.sha256(val_path.read_bytes()).hexdigest()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"candidate_images": len(val_rows), "queries": len(queries), "models": {}}
    all_rows = {}
    for name, path in (("A", args.predictions_a), ("B", args.predictions_b)):
        summary["models"][name], all_rows[name] = build_and_evaluate(
            name, path, train_rows, val_rows, args.index_dir / f"ktvic_{name.lower()}_val.npz",
            args.output_dir / f"ranks_{name}.csv", queries, metadata)
    image_ids = np.array([image_id for image_id, _, _ in queries])
    summary["a_minus_b"] = paired_bootstrap(image_ids, [row["rank"] for row in all_rows["A"]],
                                           [row["rank"] for row in all_rows["B"]], CUTOFFS)
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    (args.output_dir / "recall_summary.json").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
