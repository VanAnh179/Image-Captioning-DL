"""Check the cached CLIP region features: structure, IDs, manifest, a full re-encode of every image, batch-size independence."""

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import torch

from src.data.clip_features import (
    CLIP_CACHE_PREFIX, CLIP_FEATURE_DIM, FrozenCLIPVisionGrid, load_clip_processor, make_clip_image_loader,
)
from src.data.paths import resolve_image_root, resolve_processed_dir
from src.data.region_features import NUM_REGIONS, RegionFeatureStore, extract_split, read_manifest


def verify_split(split, processed, image_root, directory, encoder, loader, device, trace_samples, seed, expected_images):
    errors, report = [], {"split": split}
    store = RegionFeatureStore.load(directory, split, CLIP_CACHE_PREFIX)
    ids = json.loads((processed / f"{split}_ids.json").read_text(encoding="utf-8"))
    rows = read_manifest(processed / f"{split}_manifest.jsonl")

    report.update(images_expected=expected_images, ids_in_split_list=len(ids), tensors_in_cache=len(store))
    if not (len(ids) == len(store) == expected_images):
        errors.append(f"{split}: số ID danh sách={len(ids)}, số tensor={len(store)}, kỳ vọng={expected_images}")
    if store.image_ids.tolist() != ids:
        errors.append(f"{split}: ID/thứ tự trong cache khác {split}_ids.json")
    if tuple(store.features.shape[1:]) != (NUM_REGIONS, CLIP_FEATURE_DIM):
        errors.append(f"{split}: shape {tuple(store.features.shape)} không phải [N, 49, 768]")
    if [r["image_path"] for r in rows] != store.image_paths or [r["pixel_sha256"] for r in rows] != store.pixel_sha256:
        errors.append(f"{split}: đường dẫn hoặc hash pixel trong cache khác manifest")
    if not torch.isfinite(store.features.float()).all():
        errors.append(f"{split}: có NaN/Inf")
    unique = len({hashlib.sha256(store.features[i].numpy().tobytes()).digest() for i in range(len(store))})
    distinct_images = len(set(store.pixel_sha256))  # a few IDs are exact pixel duplicates and share a tensor
    report.update(distinct_tensors=unique, distinct_pixel_hashes=distinct_images)
    if unique != distinct_images:
        errors.append(f"{split}: {unique} tensor khác nhau nhưng có {distinct_images} ảnh khác nhau")

    # Re-encode EVERY image (this also re-checks every pixel hash against the manifest) and compare with the cache.
    fresh = extract_split(rows, image_root, encoder, device, batch_size=32, dtype=torch.float32, image_loader=loader)
    difference = (fresh - store.features.float()).abs()
    scale = store.features.float().abs().amax(dim=(1, 2))
    per_image = difference.amax(dim=(1, 2)) / scale
    report.update(reencoded_images=len(fresh), reencode_max_abs_diff=difference.max().item(),
                  reencode_max_relative_diff=per_image.max().item())
    if per_image.max().item() > 5e-3:  # float16 storage error is about 1e-3
        errors.append(f"{split}: sai lệch tương đối lớn nhất {per_image.max().item():.2e} giữa tensor lưu và tensor tính lại")

    picks = random.Random(seed).sample(ids, min(trace_samples, len(ids)))
    picks = sorted(set(picks) | {ids[0], ids[-1]})
    traced = 0
    for image_id in picks:
        found, _ = store.nearest_id(fresh[store.row_of[image_id]].half())
        if store.pixel_sha256[store.row_of[found]] == store.pixel_sha256[store.row_of[image_id]]:
            traced += 1
        else:
            errors.append(f"{split}: tensor tái tạo của ID {image_id} truy ngược ra ID {found} (ảnh khác)")
    report.update(traced_ok=traced, traced_total=len(picks))
    return report, errors, set(ids), rows


def batch_size_independence(rows, image_root, encoder, loader, device, count=48):
    """Features of the same images encoded with different batch sizes must agree (cuDNN TF32 used to break this)."""
    subset = rows[:count]
    outputs = {size: extract_split(subset, image_root, encoder, device, batch_size=size, dtype=torch.float32,
                                   image_loader=loader) for size in (1, 7, 32)}
    reference = outputs[32]
    return {f"batch_{size}_vs_32_max_abs_diff": (value - reference).abs().max().item() for size, value in outputs.items() if size != 32}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--features", type=Path, default=Path("artifacts/features/ktvic"))
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--trace-samples", type=int, default=50, help="Random images per split traced back by nearest ID")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--report", type=Path, default=Path("reports/clip_encoder/clip_features_check.json"))
    args = parser.parse_args()

    processed = args.processed_dir or resolve_processed_dir()
    image_root = args.image_root or resolve_image_root()
    audit = json.loads((processed / "dataset_audit.json").read_text(encoding="utf-8"))
    encoder = FrozenCLIPVisionGrid().to(args.device)
    loader = make_clip_image_loader(load_clip_processor())

    reports, errors, seen, first_rows = [], [], {}, None
    for split in args.splits:
        report, split_errors, ids, rows = verify_split(
            split, processed, image_root, args.features, encoder, loader, args.device, args.trace_samples, args.seed,
            audit["splits"][split]["images"])
        for earlier, earlier_ids in seen.items():
            if earlier_ids & ids:
                split_errors.append(f"ID trùng giữa {earlier} và {split}")
        seen[split] = ids
        first_rows = first_rows or rows
        reports.append(report)
        errors += split_errors
        print(json.dumps(report, ensure_ascii=False))

    batch_report = batch_size_independence(first_rows, image_root, encoder, loader, args.device)
    print(json.dumps(batch_report))
    if max(batch_report.values()) > 1e-3:
        errors.append(f"đặc trưng phụ thuộc kích thước batch: {batch_report}")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({"reports": reports, "batch_size_independence": batch_report, "errors": errors},
                                      ensure_ascii=False, indent=2), encoding="utf-8")
    if errors:
        print("\nLỖI:\n  " + "\n  ".join(errors), file=sys.stderr)
        sys.exit(1)
    print("\nOK: ID, thứ tự, đường dẫn, hash khớp manifest; mọi ảnh tính lại khớp cache; truy ngược đúng; không phụ thuộc batch.")


if __name__ == "__main__":
    main()
