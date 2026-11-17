"""Train A or B on shared cached regions; resume from an epoch checkpoint."""

import argparse
import random
import sys
from pathlib import Path

import torch
import yaml

from src.data.cached_caption_dataset import CachedCaptionDataset
from src.data.paths import resolve_processed_dir
from src.data.vietnamese_caption_batch import CaptionVocabulary
from src.training.caption_trainer import fit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["A", "B"], required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--epochs", type=int, default=20, help="Total target epochs, including completed epochs")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--selection-metric", choices=["cider", "val_loss"], default="cider")
    parser.add_argument("--max-len", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=4, help="Stop after this many epochs without improvement; 0 disables")
    parser.add_argument("--overfit-pairs", type=int, default=0, help="Sample this many train images, first caption only, and evaluate on those pairs")
    parser.add_argument("--dropout", type=float, help="Override model dropout, e.g. 0 for an overfit check")
    parser.add_argument("--processed-dir", type=Path)
    parser.add_argument("--features", type=Path, default=Path("artifacts/features/ktvic"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    processed = args.processed_dir or resolve_processed_dir()
    vocabulary = CaptionVocabulary.load(processed / "caption_vocabulary.json")
    default_config = "lstm_attention.yaml" if args.model == "A" else "transformer_decoder.yaml"
    config_path = args.config or Path("configs") / default_config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if args.dropout is not None:
        config["dropout"] = args.dropout
    if config["vocab_size"] != len(vocabulary.tokens):
        parser.error("Model vocab_size differs from the saved vocabulary")
    train_data = CachedCaptionDataset.load(processed, args.features, "train", vocabulary)
    if args.overfit_pairs:
        if not 1 <= args.overfit_pairs <= len(train_data.rows):
            parser.error("overfit-pairs must be between 1 and the number of training images")
        rows = random.Random(args.seed).sample(train_data.rows, args.overfit_pairs)
        train_data = CachedCaptionDataset(rows, vocabulary, train_data.store, first_caption_only=True)
        val_data = train_data
    else:
        val_data = CachedCaptionDataset.load(processed, args.features, "val", vocabulary)
    settings = {"batch_size": args.batch_size, "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay, "selection_metric": args.selection_metric,
                "max_len": args.max_len, "seed": args.seed,
                "patience": args.patience,
                "purpose": f"overfit_{args.overfit_pairs}_train_pairs" if args.overfit_pairs else "full_training"}
    fit(args.model, config, train_data, val_data, args.output, settings, args.epochs, args.device, args.resume)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    try:
        main()
    except KeyboardInterrupt:
        print("Stopped. Resume last.pt to restart after the last completed epoch.", file=sys.stderr)
        sys.exit(130)
