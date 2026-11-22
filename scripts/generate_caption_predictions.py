"""Generate validation or test captions from a trained A, B, A0 or C checkpoint."""

import argparse
import json
import time
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


class GenerationInputError(ValueError):
    """The checkpoint cannot be used with the processed data or the requested feature cache."""


def generate_predictions(checkpoint_path, split, device, processed=None, features=Path("artifacts/features/ktvic"),
                         feature_prefix=None):
    """Greedy captions, one per ID in manifest order, from the settings stored in the checkpoint.

    Returns (records, info). info has the model, epoch, parameter count, feature cache and generation seconds.
    """
    checkpoint = load_checkpoint(checkpoint_path)
    config, settings = checkpoint["config"], checkpoint["settings"]
    processed = processed or resolve_processed_dir()
    vocabulary = CaptionVocabulary.load(processed / "caption_vocabulary.json")
    if checkpoint["vocabulary"] != vocabulary.tokens:
        raise GenerationInputError("The checkpoint vocabulary differs from the processed vocabulary.")
    try:
        prefix = resolve_feature_prefix(config, feature_prefix)
    except ValueError as error:
        raise GenerationInputError(str(error)) from None
    dataset = CachedCaptionDataset.load(
        processed, features, split, vocabulary, prefix=prefix,
        feature_dim=config.get("feature_dim", FEATURE_DIM), num_regions=config.get("num_regions", NUM_REGIONS),
    )
    model = build_model(checkpoint["model_name"], config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    use_cuda = torch.device(device).type == "cuda"
    # The first pass warms up CUDA and is not timed; the second pass is the reported time and must repeat the captions.
    predictions = predict_captions(model, dataset, settings["batch_size"], device, settings["max_len"])
    if use_cuda:
        torch.cuda.synchronize()
    started = time.perf_counter()
    repeated = predict_captions(model, dataset, settings["batch_size"], device, settings["max_len"])
    if use_cuda:
        torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    records = [{"image_id": image_id, "caption": caption} for image_id, caption in predictions]
    info = {"model_name": checkpoint["model_name"], "epoch": checkpoint["epoch"], "config": config,
            "batch_size": settings["batch_size"], "max_len": settings["max_len"], "feature_prefix": prefix,
            "parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
            "generation_seconds": round(seconds, 2), "generation_repeat_identical": predictions == repeated,
            "generation_device": str(device),
            "generation_gpu": torch.cuda.get_device_name(0) if use_cuda else None, "torch": str(torch.__version__)}
    return records, info


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

    try:
        records, _ = generate_predictions(args.checkpoint, args.split, args.device, args.processed_dir,
                                          args.features, args.feature_prefix)
    except GenerationInputError as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Generated {len(records)} captions in {args.output}")


if __name__ == "__main__":
    main()
