"""Generate captions and aligned per-token logits/attention from an LSTM checkpoint."""

import argparse
import json
import sys
from pathlib import Path

import torch

from src.data.paths import resolve_processed_dir
from src.data.region_features import RegionFeatureStore, read_manifest
from src.data.vietnamese_caption_batch import CaptionVocabulary
from src.models.lstm_attention_decoder import LSTMAttentionCaptionDecoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--image-ids", type=int, nargs="+", help="Omit to generate for the complete split")
    parser.add_argument("--processed-dir", type=Path)
    parser.add_argument("--features", type=Path, default=Path("artifacts/features/ktvic"))
    parser.add_argument("--output", type=Path, default=Path("runs/lstm_predictions"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-len", type=int, default=40)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.max_len <= 0:
        parser.error("batch-size and max-len must be positive")

    processed = args.processed_dir or resolve_processed_dir()
    vocabulary = CaptionVocabulary.load(processed / "caption_vocabulary.json")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["vocabulary"] != vocabulary.tokens:
        raise ValueError("Checkpoint vocabulary differs from the processed vocabulary")
    model = LSTMAttentionCaptionDecoder(**checkpoint["config"]).to(args.device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    store = RegionFeatureStore.load(args.features, args.split)
    split_ids = [row["image_id"] for row in read_manifest(processed / f"{args.split}_manifest.jsonl")]
    ids = args.image_ids if args.image_ids is not None else split_ids
    if len(set(ids)) != len(ids) or not set(ids).issubset(split_ids):
        raise ValueError("Requested image IDs must be unique and belong to the selected split")
    missing = set(ids) - set(store.row_of)
    if missing:
        raise ValueError(f"Feature cache is missing image IDs: {sorted(missing)[:10]}")
    predictions, records = [], []
    for start in range(0, len(ids), args.batch_size):
        batch_ids = ids[start:start + args.batch_size]
        output = model.generate(store.get_batch(batch_ids).to(args.device), max_len=args.max_len)
        output = {key: value.cpu() for key, value in output.items()}
        for row, image_id in enumerate(batch_ids):
            length = int(output["lengths"][row])
            token_ids = output["token_ids"][row, :length]
            caption = vocabulary.decode(token_ids)
            predictions.append({"image_id": image_id, "caption": caption})
            records.append({**store.trace(image_id), "caption": caption, "token_ids": token_ids,
                            "tokens": [vocabulary.tokens[i] for i in token_ids.tolist()],
                            "logits": output["logits"][row, :length],
                            "attention": output["attention"][row, :length],
                            "finished": bool(output["finished"][row])})
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"{args.split}_predictions.json").write_text(
        json.dumps(predictions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    torch.save({"checkpoint": str(args.checkpoint), "split": args.split,
                "checkpoint_purpose": checkpoint.get("purpose", "unspecified"),
                "max_len": args.max_len, "grid_shape": [7, 7],
                "layout": "attention[t, row * 7 + column] aligns with token_ids[t] and logits[t]",
                "records": records}, args.output / f"{args.split}_attention.pt")
    print(f"Generated {len(records)} captions in {args.output}; EOS reached for {sum(r['finished'] for r in records)} images")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    main()
