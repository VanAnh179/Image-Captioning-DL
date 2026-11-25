"""Build a train-fitted TF-IDF index of A's generated captions for the validation images, or for validation and test together."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

from src.data.region_features import read_manifest
from src.evaluation.caption_scorer import load_predictions
from src.retrieval.caption_search import CaptionSearchIndex, pool_digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, nargs="+", required=True,
                        help="A's val_predictions.json (and A's test predictions with --candidate-splits val test)")
    parser.add_argument("--candidate-splits", nargs="+", choices=("val", "test"), default=["val"],
                        help="images the search can return; val test makes a pool of both")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed/ktvic"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/retrieval/ktvic_a_val.npz"))
    args = parser.parse_args()
    splits = list(dict.fromkeys(args.candidate_splits))
    train = args.processed_dir / "train_manifest.jsonl"
    manifests = {split: args.processed_dir / f"{split}_manifest.jsonl" for split in splits}
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    metadata = {"train_manifest_sha256": digest(train)}
    metadata.update({f"{split}_manifest_sha256": digest(path) for split, path in manifests.items()})
    name = "+".join(splits)
    if len(splits) > 1:
        metadata[f"{name}_manifest_sha256"] = pool_digest([metadata[f"{split}_manifest_sha256"] for split in splits])
    hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in args.predictions]
    metadata["predictions_sha256"] = hashes[0] if len(hashes) == 1 else hashes
    predictions = []
    for path in args.predictions:
        predictions.extend(load_predictions(path))
    candidates = [row for split in splits for row in read_manifest(manifests[split])]
    index = CaptionSearchIndex.build(read_manifest(train), predictions, candidates, metadata, candidate_split=name)
    index.save(args.output)
    print(json.dumps(index.metadata, ensure_ascii=False, indent=2))
    print(f"Index saved to {args.output}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
