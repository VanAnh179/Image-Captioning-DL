"""Fit the image-to-text projection on KTVIC train and index val or test for offline search."""

import argparse
import hashlib
import os
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch
from torch.nn import functional as F

from src.data.paths import resolve_processed_dir
from src.data.region_features import RegionFeatureStore, read_manifest
from src.retrieval.visual_search import DEFAULT_INDEX_DIR, FORMAT_VERSION, TEXT_MODEL, TEXT_REVISION, MiniLMEncoder


def load_split(processed_dir, feature_dir, split, device):
    manifest = processed_dir / f"{split}_manifest.jsonl"
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    rows = read_manifest(manifest)
    ids = [row["image_id"] for row in rows]
    store = RegionFeatureStore.load(feature_dir, split)
    if len(ids) != len(store) or set(ids) != store.row_of.keys() or store.meta.get("manifest_sha256") != digest:
        raise ValueError(f"Cache ResNet-50 và manifest của tập {split} không khớp.")
    pooled = []
    for start in range(0, len(ids), 256):
        regions = store.get_batch(ids[start:start + 256]).to(device)
        pooled.append(torch.cat((regions.mean(dim=1), regions.amax(dim=1)), dim=1).clamp_min(0).sqrt())
    return rows, digest, ids, torch.cat(pooled)


def fit_projection(image_features, captions, encoder):
    # One target per image, averaging its five train references in the MiniLM space.
    unique = list(dict.fromkeys(captions))
    positions = {caption: position for position, caption in enumerate(unique)}
    vectors = encoder.encode(unique)
    rows = torch.tensor([positions[caption] for caption in captions], device=vectors.device)
    embeddings = vectors[rows].reshape(len(image_features), 5, -1)
    text_center = embeddings.mean(dim=(0, 1))
    targets = F.normalize(embeddings.mean(dim=1) - text_center, dim=1)
    image_center = image_features.mean(dim=0)
    image_scale = image_features.std(dim=0).clamp_min(0.025)
    images = F.normalize((image_features - image_center) / image_scale, dim=1)

    # Dual ridge regression with a coefficient of 1.0 and unit-norm image features.
    gram = images @ images.T
    gram.diagonal().add_(1.0)
    projection = images.T @ torch.linalg.solve(gram, targets)
    return image_center, image_scale, text_center, projection


def save_index(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(dir=path.parent, delete=False) as file:
            temporary = Path(file.name)
            torch.save(data, file)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path)
    parser.add_argument("--features", type=Path, default=Path("artifacts/features/ktvic"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--splits", nargs="+", choices=("val", "test"), default=["val"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    try:
        processed = args.processed_dir or resolve_processed_dir()
        train_rows, train_digest, _, train_images = load_split(processed, args.features, "train", args.device)
        if not all(len(row["captions"]) == 5 for row in train_rows):
            raise ValueError("Mỗi ảnh train phải có đúng năm caption tham chiếu.")
        encoder = MiniLMEncoder(args.device)
        captions = [caption for row in train_rows for caption in row["captions"]]
        print(f"Đang học phép chiếu từ {len(train_rows)} ảnh và {len(captions)} câu train…", flush=True)
        center, scale, text_center, projection = fit_projection(train_images, captions, encoder)
        for split in dict.fromkeys(args.splits):
            _, digest, ids, images = load_split(processed, args.features, split, args.device)
            normalized = F.normalize((images - center) / scale, dim=1)
            vectors = F.normalize(normalized @ projection, dim=1)
            path = args.output_dir / f"ktvic_{split}.pt"
            save_index(path, {"format_version": FORMAT_VERSION, "text_model": TEXT_MODEL,
                              "text_revision": TEXT_REVISION, "split": split, "manifest_sha256": digest,
                              "train_manifest_sha256": train_digest, "image_ids": torch.tensor(ids),
                              "vectors": vectors.cpu(), "text_center": text_center.cpu()})
            print(f"Đã lưu {len(ids)} vector ảnh {split} vào {path}.", flush=True)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Không tạo được chỉ mục ảnh: {error}\n")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    main()
