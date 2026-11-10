"""Check cached region features against the split lists and trace sampled tensors back to their images."""

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import torch

from src.data.paths import resolve_image_root, resolve_processed_dir
from src.data.region_features import (
    FEATURE_DIM, FORMAT_VERSION, NUM_REGIONS, WEIGHTS_NAME,
    FrozenResNet50Grid, RegionFeatureStore, load_image, read_manifest,
)


def verify_split(split, processed, image_root, directory, encoder, device, samples, seed, expected_images):
    errors, report = [], {"split": split}
    store = RegionFeatureStore.load(directory, split)
    ids = json.loads((processed / f"{split}_ids.json").read_text(encoding="utf-8"))
    rows = read_manifest(processed / f"{split}_manifest.jsonl")

    report["images_expected"] = expected_images
    report["ids_in_split_list"] = len(ids)
    report["tensors_in_cache"] = len(store)
    if not (len(ids) == len(store) == expected_images):
        errors.append(f"{split}: số ID danh sách={len(ids)}, số tensor={len(store)}, kỳ vọng={expected_images}")
    if store.image_ids.tolist() != ids:
        errors.append(f"{split}: ID/thứ tự trong cache khác {split}_ids.json")
    if [r["image_id"] for r in rows] != ids:
        errors.append(f"{split}: ID/thứ tự trong manifest khác {split}_ids.json")
    manifest_hash = hashlib.sha256((processed / f"{split}_manifest.jsonl").read_bytes()).hexdigest()
    if store.meta.get("manifest_sha256") != manifest_hash:
        errors.append(f"{split}: hash manifest khác bản dùng để trích đặc trưng")
    if (store.meta.get("format_version") != FORMAT_VERSION or store.meta.get("split") != split
            or store.meta.get("weights") != WEIGHTS_NAME):
        errors.append(f"{split}: metadata phiên bản cache/tập dữ liệu/trọng số ResNet không khớp")
    if store.features.shape[1:] != (NUM_REGIONS, FEATURE_DIM):
        errors.append(f"{split}: shape {tuple(store.features.shape)} không phải [N, 49, 2048]")
    if [r["image_path"] for r in rows] != store.image_paths or [r["pixel_sha256"] for r in rows] != store.pixel_sha256:
        errors.append(f"{split}: đường dẫn hoặc hash pixel trong cache khác manifest")
    if not torch.isfinite(store.features.float()).all():
        errors.append(f"{split}: có NaN/Inf")
    unique = len({hashlib.sha256(store.features[i].numpy().tobytes()).digest() for i in range(len(store))})
    distinct_images = len(set(store.pixel_sha256))  # a few IDs are exact pixel duplicates and share a tensor
    report["distinct_tensors"], report["distinct_pixel_hashes"] = unique, distinct_images
    if unique != distinct_images:
        errors.append(f"{split}: {unique} tensor khác nhau nhưng có {distinct_images} ảnh khác nhau")

    picks = random.Random(seed).sample(ids, min(samples, len(ids)))
    picks = sorted(set(picks) | {ids[0], ids[-1]})
    worst, traced = 0.0, 0
    for image_id in picks:
        info = store.trace(image_id)
        tensor, digest = load_image(image_root / info["image_path"])
        fresh = encoder(tensor.unsqueeze(0).to(device))[0].float().cpu()
        cached = store.get_batch([image_id])[0]
        scale = max(cached.abs().max().item(), 1e-8)
        worst = max(worst, (fresh - cached).abs().max().item() / scale)
        found, distance = store.nearest_id(fresh)
        if digest != info["pixel_sha256"]:
            errors.append(f"{split}: ID {image_id} hash pixel của file ảnh khác bản ghi")
        if store.pixel_sha256[store.row_of[found]] != info["pixel_sha256"]:
            errors.append(f"{split}: tensor tái tạo của ID {image_id} truy ngược ra ID {found} (ảnh khác)")
        else:
            traced += 1
    report.update(traced_ok=traced, traced_total=len(picks), max_relative_error=worst)
    if worst > 5e-3:  # float16 storage error is ~1e-3
        errors.append(f"{split}: sai lệch tương đối {worst:.2e} giữa tensor lưu và tensor tính lại")
    return report, errors, set(ids)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--features", type=Path, default=Path("artifacts/features/ktvic"))
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--samples", type=int, default=25, help="Random images per split to trace back")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--report", type=Path, default=Path("runs/region_features_check/report.json"))
    args = parser.parse_args()

    processed = args.processed_dir or resolve_processed_dir()
    image_root = args.image_root or resolve_image_root()
    audit = json.loads((processed / "dataset_audit.json").read_text(encoding="utf-8"))
    encoder = FrozenResNet50Grid().to(args.device)

    reports, all_errors, seen = [], [], {}
    for split in args.splits:
        report, errors, ids = verify_split(split, processed, image_root, args.features, encoder, args.device,
                                           args.samples, args.seed, audit["splits"][split]["images"])
        for earlier, earlier_ids in seen.items():
            if earlier_ids & ids:
                errors.append(f"ID trùng giữa {earlier} và {split}")
        seen[split] = ids
        reports.append(report)
        all_errors += errors
        print(json.dumps(report, ensure_ascii=False))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({"reports": reports, "errors": all_errors}, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    if all_errors:
        print("\nLỖI:\n  " + "\n  ".join(all_errors), file=sys.stderr)
        sys.exit(1)
    print("\nOK: số ID khớp từng tập, thứ tự/đường dẫn/hash khớp manifest, tensor truy ngược đúng ảnh.")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    main()
