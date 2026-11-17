"""Measure what cuDNN TF32 does to the ResNet-50 features of model B and to B's val score.

Compares, on the val split with the trained B checkpoint: (a) the cached features B was trained on, (b) the same features
recomputed with the default settings (TF32 on), (c) recomputed in full fp32 (TF32 off, as the CLIP encoder does).
"""

import argparse
import json
from pathlib import Path

import torch

from scripts.check_clip_pipeline import greedy_decode
from src.data.paths import resolve_image_root, resolve_processed_dir
from src.data.region_features import FrozenResNet50Grid, RegionFeatureStore, extract_split, read_manifest
from src.data.vietnamese_caption_batch import CaptionVocabulary
from src.evaluation.caption_scorer import score_captions
from src.models.transformer_decoder import TransformerCaptionDecoder


@torch.no_grad()
def caption_val(model, features, ids, vocabulary, device, max_len, batch_size=32):
    predictions = []
    for start in range(0, len(ids), batch_size):
        tokens = greedy_decode(model, features[start:start + batch_size].float().to(device), max_len)
        predictions += [vocabulary.decode(row) for row in tokens.tolist()]
    return list(zip(ids, predictions))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/checkpoints/ktvic/B_seed42_best.pt"))
    parser.add_argument("--features", type=Path, default=Path("artifacts/features/ktvic"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--max-len", type=int, default=40)
    parser.add_argument("--output", type=Path, default=Path("reports/clip_encoder/tf32_effect_on_B.json"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    processed, image_root = resolve_processed_dir(), resolve_image_root()
    rows = read_manifest(processed / f"{args.split}_manifest.jsonl")
    ids = [row["image_id"] for row in rows]
    references = {row["image_id"]: row["captions"] for row in rows}
    vocabulary = CaptionVocabulary.load(processed / "caption_vocabulary.json")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = TransformerCaptionDecoder(**checkpoint["config"]).to(args.device)
    model.load_state_dict(checkpoint["model_state"])

    cached = RegionFeatureStore.load(args.features, args.split).get_batch(ids)
    encoder = FrozenResNet50Grid().to(args.device)
    tf32_default = torch.backends.cudnn.allow_tf32
    on = extract_split(rows, image_root, encoder, args.device, dtype=torch.float16)
    with torch.backends.cudnn.flags(allow_tf32=False):
        off = extract_split(rows, image_root, encoder, args.device, dtype=torch.float16)

    def relative(a, b):
        return ((a.float() - b.float()).abs().amax(dim=(1, 2)) / b.float().abs().amax(dim=(1, 2))).max().item()

    report = {"cudnn_allow_tf32_default": tf32_default, "images": len(ids),
              "feature_max_relative_diff": {
                  "recomputed_default_vs_cache": relative(on, cached),
                  "recomputed_fp32_vs_cache": relative(off, cached),
                  "recomputed_fp32_vs_recomputed_default": relative(off, on)},
              "val_scores": {}, "captions_different_from_cache": {}}
    cache_captions = None
    for name, features in (("cache_used_to_train_B", cached), ("recomputed_default_tf32_on", on), ("recomputed_fp32_tf32_off", off)):
        predictions = caption_val(model, features, ids, vocabulary, args.device, args.max_len)
        scores = score_captions(predictions, references)["scores"]
        report["val_scores"][name] = scores
        if cache_captions is None:
            cache_captions = predictions
        report["captions_different_from_cache"][name] = sum(a[1] != b[1] for a, b in zip(predictions, cache_captions))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
