"""Generate validation or test captions from a trained A, B, A0 or C checkpoint."""

import argparse
import json
from pathlib import Path

import torch

from src.data.cached_caption_dataset import CachedCaptionDataset
from src.data.paths import resolve_processed_dir
from src.data.region_features import CACHE_PREFIX, FEATURE_DIM, NUM_REGIONS
from src.data.vietnamese_caption_batch import CaptionVocabulary
from src.training.caption_trainer import build_model, load_checkpoint, predict_captions


def resolve_feature_prefix(config, explicit=None):
    """Feature cache prefix for the checkpoint's regions. An explicit prefix always wins."""
    if explicit:
        return explicit
    feature_dim = config.get("feature_dim", FEATURE_DIM)
    if feature_dim == FEATURE_DIM:
        return CACHE_PREFIX
    from src.data.clip_features import CLIP_CACHE_PREFIX, CLIP_FEATURE_DIM  # imports transformers, only needed for C

    if feature_dim == CLIP_FEATURE_DIM:
        return CLIP_CACHE_PREFIX
    raise ValueError(f"Cannot infer the feature cache for feature_dim={feature_dim}; pass --feature-prefix.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--processed-dir", type=Path)
    parser.add_argument("--features", type=Path, default=Path("artifacts/features/ktvic"))
    parser.add_argument("--feature-prefix", default=None,
                        help="Feature cache prefix. Default: inferred from the checkpoint "
                             "(resnet50_7x7 for A, B and A0; clip_vit_b32_7x7 for C).")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    checkpoint = load_checkpoint(args.checkpoint)
    config, settings = checkpoint["config"], checkpoint["settings"]
    processed = args.processed_dir or resolve_processed_dir()
    vocabulary = CaptionVocabulary.load(processed / "caption_vocabulary.json")
    if checkpoint["vocabulary"] != vocabulary.tokens:
        parser.error("The checkpoint vocabulary differs from the processed vocabulary.")
    try:
        prefix = resolve_feature_prefix(config, args.feature_prefix)
    except ValueError as error:
        parser.error(str(error))
    dataset = CachedCaptionDataset.load(
        processed, args.features, args.split, vocabulary, prefix=prefix,
        feature_dim=config.get("feature_dim", FEATURE_DIM), num_regions=config.get("num_regions", NUM_REGIONS),
    )
    model = build_model(checkpoint["model_name"], config).to(args.device)
    model.load_state_dict(checkpoint["model_state"])
    predictions = predict_captions(model, dataset, settings["batch_size"], args.device, settings["max_len"])
    records = [{"image_id": image_id, "caption": caption} for image_id, caption in predictions]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Generated {len(records)} captions in {args.output}")


if __name__ == "__main__":
    main()
