"""Encode every KTVIC image with a frozen CLIP ViT-B/32 into a 7x7 grid (49 regions x 768) keyed by image ID."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
import transformers

from src.data.clip_features import (
    CLIP_CACHE_PREFIX, CLIP_FEATURE_DIM, CLIP_MODEL, CLIP_REVISION, FORMAT_VERSION,
    FrozenCLIPVisionGrid, load_clip_processor, make_clip_image_loader,
)
from src.training.run_log import tee_log
from src.data.paths import FEATURE_DIR, LOG_DIR, resolve_image_root, resolve_processed_dir
from src.data.region_features import GRID, NUM_REGIONS, extract_split, feature_path, read_manifest


def sha256_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, help="Folder with *_manifest.jsonl (auto-detected)")
    parser.add_argument("--image-root", type=Path, help="Folder with train-images/ (auto-detected)")
    parser.add_argument("--output", type=Path, default=FEATURE_DIR)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    args = parser.parse_args()

    processed = args.processed_dir or resolve_processed_dir()
    image_root = args.image_root or resolve_image_root()
    dtype = getattr(torch, args.dtype)
    processor = load_clip_processor()
    encoder = FrozenCLIPVisionGrid().to(args.device)
    args.output.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        manifest = processed / f"{split}_manifest.jsonl"
        rows = read_manifest(manifest)
        ids = json.loads((processed / f"{split}_ids.json").read_text(encoding="utf-8"))
        if [row["image_id"] for row in rows] != ids:
            raise ValueError(f"{split}: thứ tự ID trong manifest khác {split}_ids.json")
        started = time.time()
        features = extract_split(rows, image_root, encoder, args.device, args.batch_size, args.workers, dtype,
                                 image_loader=make_clip_image_loader(processor))
        assert features.shape == (len(rows), NUM_REGIONS, CLIP_FEATURE_DIM), features.shape
        if not torch.isfinite(features.float()).all():
            raise ValueError(f"{split}: đặc trưng có NaN/Inf")
        payload = {
            "image_ids": torch.tensor(ids, dtype=torch.long),
            "features": features,
            "image_paths": [row["image_path"] for row in rows],
            "pixel_sha256": [row["pixel_sha256"] for row in rows],
            "meta": {
                "format_version": FORMAT_VERSION, "split": split,
                "encoder": "CLIPVisionModel last_hidden_state[:, 1:] (class token dropped), frozen, eval()",
                "weights": CLIP_MODEL, "weights_revision": CLIP_REVISION,
                "preprocessing": processor.to_dict(),
                "input_size": processor.crop_size["height"], "grid": GRID, "feature_dim": CLIP_FEATURE_DIM,
                "layout": "[image, region, channel]; region = row * 7 + column",
                "dtype": args.dtype,
                "manifest": manifest.name, "manifest_sha256": sha256_file(manifest),
                "torch": str(torch.__version__), "transformers": str(transformers.__version__),
                "pixel_hash_checked_against_manifest": True,
            },
        }
        target = feature_path(args.output, split, CLIP_CACHE_PREFIX)
        torch.save(payload, target)
        print(f"{split}: {len(rows)} ảnh -> {target} "
              f"({target.stat().st_size / 2**20:.0f} MiB, {time.time() - started:.0f}s)")


if __name__ == "__main__":
    with tee_log(LOG_DIR / "extract_clip_features.log"):
        main()
