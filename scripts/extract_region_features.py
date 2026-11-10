"""Encode every KTVIC image with a frozen ResNet-50 into a 7x7 grid (49 regions x 2048) keyed by image ID."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
import torchvision

from src.data.paths import resolve_image_root, resolve_processed_dir
from src.data.region_features import (
    FEATURE_DIM, FORMAT_VERSION, GRID, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, NUM_REGIONS,
    WEIGHTS_NAME, FrozenResNet50Grid, extract_split, feature_path, read_manifest,
)


def sha256_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, help="Folder with *_manifest.jsonl (auto-detected)")
    parser.add_argument("--image-root", type=Path, help="Folder with train-images/ (auto-detected)")
    parser.add_argument("--output", type=Path, default=Path("artifacts/features/ktvic"))
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    args = parser.parse_args()

    processed = args.processed_dir or resolve_processed_dir()
    image_root = args.image_root or resolve_image_root()
    dtype = getattr(torch, args.dtype)
    encoder = FrozenResNet50Grid().to(args.device)
    args.output.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        manifest = processed / f"{split}_manifest.jsonl"
        rows = read_manifest(manifest)
        ids = json.loads((processed / f"{split}_ids.json").read_text(encoding="utf-8"))
        if [row["image_id"] for row in rows] != ids:
            raise ValueError(f"{split}: thứ tự ID trong manifest khác {split}_ids.json")
        started = time.time()
        features = extract_split(rows, image_root, encoder, args.device, args.batch_size, args.workers, dtype)
        assert features.shape == (len(rows), NUM_REGIONS, FEATURE_DIM), features.shape
        if not torch.isfinite(features.float()).all():
            raise ValueError(f"{split}: đặc trưng có NaN/Inf")
        payload = {
            "image_ids": torch.tensor(ids, dtype=torch.long),
            "features": features,
            "image_paths": [row["image_path"] for row in rows],
            "pixel_sha256": [row["pixel_sha256"] for row in rows],
            "meta": {
                "format_version": FORMAT_VERSION, "split": split,
                "encoder": "torchvision resnet50 (conv1..layer4), frozen, eval()",
                "weights": WEIGHTS_NAME, "input_size": IMAGE_SIZE, "grid": GRID, "feature_dim": FEATURE_DIM,
                "layout": "[image, region, channel]; region = row * 7 + column",
                "dtype": args.dtype, "normalize_mean": list(IMAGENET_MEAN), "normalize_std": list(IMAGENET_STD),
                "manifest": manifest.name, "manifest_sha256": sha256_file(manifest),
                "torch": str(torch.__version__), "torchvision": str(torchvision.__version__),
                "pixel_hash_checked_against_manifest": True,
            },
        }
        target = feature_path(args.output, split)
        torch.save(payload, target)
        print(f"{split}: {len(rows)} ảnh -> {target} "
              f"({target.stat().st_size / 2**20:.0f} MiB, {time.time() - started:.0f}s)")


if __name__ == "__main__":
    main()
