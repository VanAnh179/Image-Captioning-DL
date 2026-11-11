"""Replace the end of real val captions and confirm the decoder's predictions for the beginning do not move (eval mode)."""

import argparse
import json
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence

from src.data.paths import resolve_processed_dir
from src.data.region_features import RegionFeatureStore, read_manifest
from src.data.vietnamese_caption_batch import BOS, PAD, CaptionVocabulary
from src.models import transformer_decoder as td


def build_pairs(rows, vocabulary, generator, min_words=4):
    """(image_id, input ids, changed input ids, first changed input position) for every val image."""
    pairs = []
    for row in rows:
        words = vocabulary.encode(row["captions"][0])
        if len(words) < min_words:
            continue
        keep = len(words) // 2  # words kept; the rest of the sentence is replaced
        new_tail = torch.randint(4, len(vocabulary.tokens), (len(words) - keep,), generator=generator)
        for n, old in enumerate(words[keep:]):  # make sure every replaced word really differs
            if new_tail[n] == old:
                new_tail[n] = 4 + (int(old) - 3) % (len(vocabulary.tokens) - 4)
        original = torch.tensor([BOS, *words])
        changed = torch.cat([original[:keep + 1], new_tail])
        pairs.append((row["image_id"], original, changed, keep + 1))
    return pairs


@torch.no_grad()
def run(model, store, pairs, batch_size):
    """Head positions: logits of original vs changed sentence. Returns worst abs diff, mismatched argmax, tail diff."""
    head_diff, head_argmax_mismatch, head_total, tail_changed, tail_total = 0.0, 0, 0, 0, 0
    for start in range(0, len(pairs), batch_size):
        chunk = pairs[start:start + batch_size]
        regions = store.get_batch([p[0] for p in chunk])
        original = pad_sequence([p[1] for p in chunk], batch_first=True, padding_value=PAD)
        changed = pad_sequence([p[2] for p in chunk], batch_first=True, padding_value=PAD)
        a, b = model(regions, original), model(regions, changed)
        for n, (_, ids, _, first_changed) in enumerate(chunk):
            head_diff = max(head_diff, (a[n, :first_changed] - b[n, :first_changed]).abs().max().item())
            head_argmax_mismatch += int((a[n, :first_changed].argmax(-1) != b[n, :first_changed].argmax(-1)).sum())
            head_total += first_changed
            tail = slice(first_changed, len(ids))
            tail_changed += int((a[n, tail].argmax(-1) != b[n, tail].argmax(-1)).sum())
            tail_total += len(ids) - first_changed
    return {"head_positions": head_total, "head_max_abs_logit_diff": head_diff,
            "head_argmax_changed": head_argmax_mismatch, "tail_positions": tail_total, "tail_argmax_changed": tail_changed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path)
    parser.add_argument("--features", type=Path, default=Path("artifacts/features/ktvic"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("runs/transformer_decoder_check/report.json"))
    args = parser.parse_args()

    processed = args.processed_dir or resolve_processed_dir()
    vocabulary = CaptionVocabulary.load(processed / "caption_vocabulary.json")
    rows = read_manifest(processed / f"{args.split}_manifest.jsonl")
    store = RegionFeatureStore.load(args.features, args.split)
    generator = torch.Generator().manual_seed(args.seed)
    pairs = build_pairs(rows, vocabulary, generator)

    torch.manual_seed(args.seed)
    model = td.TransformerCaptionDecoder(len(vocabulary.tokens)).eval()
    report = {"split": args.split, "images": len(pairs), "seed": args.seed,
              "parameters": sum(p.numel() for p in model.parameters()),
              "result": run(model, store, pairs, args.batch_size)}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    ok = report["result"]["head_argmax_changed"] == 0 and report["result"]["head_max_abs_logit_diff"] < 1e-4
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
