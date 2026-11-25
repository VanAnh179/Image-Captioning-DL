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
from src.retrieval.caption_search import pool_digest
from src.retrieval.visual_search import DEFAULT_INDEX_DIR, FORMAT_VERSION, TEXT_MODEL, TEXT_REVISION, MiniLMEncoder


# name -> (cache prefix, feature size, floor for the per-feature scale). ResNet cells follow ReLU, so the square root
# tames its heavy tail; CLIP tokens are signed and used as they are.
FEATURES = {"resnet50": ("resnet50_7x7", 2048, 0.025), "clip": ("clip_vit_b32_7x7", 768, 1e-3)}


def load_split(processed_dir, feature_dir, split, device, names=("resnet50",)):
    """Pooled features per name: the mean and the maximum over the 49 regions of each cached image."""
    manifest = processed_dir / f"{split}_manifest.jsonl"
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    rows = read_manifest(manifest)
    ids = [row["image_id"] for row in rows]
    parts = []
    for name in names:
        prefix, size, _ = FEATURES[name]
        store = RegionFeatureStore.load(feature_dir, split, prefix=prefix, feature_dim=size)
        if len(ids) != len(store) or set(ids) != store.row_of.keys() or store.meta.get("manifest_sha256") != digest:
            raise ValueError(f"Cache {name} và manifest của tập {split} không khớp.")
        pooled = []
        for start in range(0, len(ids), 256):
            regions = store.get_batch(ids[start:start + 256]).float().to(device)
            vector = torch.cat((regions.mean(dim=1), regions.amax(dim=1)), dim=1)
            pooled.append(vector.clamp_min(0).sqrt() if name == "resnet50" else vector)
        parts.append(torch.cat(pooled))
    return rows, digest, ids, parts


def normalize_parts(parts, state):
    """Standardize each feature with train statistics, unit-normalize it, then concatenate and normalize again."""
    return F.normalize(torch.cat([F.normalize((part - center) / scale, dim=1)
                                  for part, (center, scale) in zip(parts, state)], dim=1), dim=1)


def fit_projection(image_parts, captions, encoder, names=("resnet50",)):
    # One target per image, averaging its five train references in the MiniLM space.
    unique = list(dict.fromkeys(captions))
    positions = {caption: position for position, caption in enumerate(unique)}
    vectors = encoder.encode(unique)
    rows = torch.tensor([positions[caption] for caption in captions], device=vectors.device)
    embeddings = vectors[rows].reshape(len(image_parts[0]), 5, -1)
    text_center = embeddings.mean(dim=(0, 1))
    targets = F.normalize(embeddings.mean(dim=1) - text_center, dim=1)
    state = [(part.mean(dim=0), part.std(dim=0).clamp_min(FEATURES[name][2])) for part, name in zip(image_parts, names)]
    images = normalize_parts(image_parts, state)

    # Dual ridge regression with a coefficient of 1.0 and unit-norm image features.
    gram = images @ images.T
    gram.diagonal().add_(1.0)
    projection = images.T @ torch.linalg.solve(gram, targets)
    return state, text_center, projection


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
    parser.add_argument("--with-clip", action="store_true",
                        help="add pooled CLIP ViT-B/32 features to the ResNet-50 ones (needs the CLIP caches of train and the split)")
    parser.add_argument("--splits", nargs="+", choices=("val", "test"), default=["val"])
    parser.add_argument("--combine", action="store_true",
                        help="also save one index over val and test together (ktvic_val_test.pt); needs both splits")
    parser.add_argument("--text-model-dir", type=Path, help="folder made by scripts.export_text_model")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    try:
        processed = args.processed_dir or resolve_processed_dir()
        names = ("resnet50", "clip") if args.with_clip else ("resnet50",)
        train_rows, train_digest, _, train_images = load_split(processed, args.features, "train", args.device, names)
        if not all(len(row["captions"]) == 5 for row in train_rows):
            raise ValueError("Mỗi ảnh train phải có đúng năm caption tham chiếu.")
        encoder = MiniLMEncoder(args.device, args.text_model_dir)
        captions = [caption for row in train_rows for caption in row["captions"]]
        print(f"Đang học phép chiếu từ {len(train_rows)} ảnh và {len(captions)} câu train…", flush=True)
        state, text_center, projection = fit_projection(train_images, captions, encoder, names)
        if args.combine and not {"val", "test"} <= set(args.splits):
            raise ValueError("--combine cần cả hai tập: --splits val test")
        parts = {}
        for split in dict.fromkeys(args.splits):
            _, digest, ids, images = load_split(processed, args.features, split, args.device, names)
            vectors = F.normalize(normalize_parts(images, state) @ projection, dim=1)
            parts[split] = (digest, ids, vectors.cpu())

        def save(name, split, digest, ids, vectors):
            path = args.output_dir / f"ktvic_{name}.pt"
            save_index(path, {"format_version": FORMAT_VERSION, "text_model": TEXT_MODEL,
                              "text_revision": TEXT_REVISION, "split": split, "manifest_sha256": digest,
                              "train_manifest_sha256": train_digest, "image_features": list(names),
                              "image_ids": torch.tensor(ids), "vectors": vectors, "text_center": text_center.cpu()})
            print(f"Đã lưu {len(ids)} vector ảnh {split} vào {path}.", flush=True)

        for split, (digest, ids, vectors) in parts.items():
            save(split, split, digest, ids, vectors)
        if args.combine:
            ids = parts["val"][1] + parts["test"][1]
            if len(set(ids)) != len(ids):
                raise ValueError("Val và test có ID ảnh trùng nhau.")
            save("val_test", "val+test", pool_digest([parts["val"][0], parts["test"][0]]), ids,
                 torch.cat((parts["val"][2], parts["test"][2])))
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Không tạo được chỉ mục ảnh: {error}\n")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    main()
