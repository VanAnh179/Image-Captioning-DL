"""Build a train-fitted TF-IDF index of A's generated validation captions."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

from src.data.region_features import read_manifest
from src.evaluation.caption_scorer import load_predictions
from src.retrieval.caption_search import CaptionSearchIndex


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True, help="A's val_predictions.json")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed/ktvic"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/retrieval/ktvic_a_val.npz"))
    args = parser.parse_args()
    train = args.processed_dir / "train_manifest.jsonl"
    val = args.processed_dir / "val_manifest.jsonl"
    metadata = {"train_manifest_sha256": hashlib.sha256(train.read_bytes()).hexdigest(),
                "val_manifest_sha256": hashlib.sha256(val.read_bytes()).hexdigest(),
                "predictions_sha256": hashlib.sha256(args.predictions.read_bytes()).hexdigest()}
    index = CaptionSearchIndex.build(read_manifest(train), load_predictions(args.predictions), read_manifest(val), metadata)
    index.save(args.output)
    print(json.dumps(index.metadata, ensure_ascii=False, indent=2))
    print(f"Index saved to {args.output}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
