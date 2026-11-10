"""Locate the KTVIC folders laid out as in the README."""

from pathlib import Path


def resolve_processed_dir(root="."):
    """Folder holding {train,val,test}_ids.json, manifests and the vocabulary."""
    path = Path(root) / "data/processed/ktvic"
    if not (path / "train_manifest.jsonl").is_file():
        raise FileNotFoundError(f"Không thấy train_manifest.jsonl trong {path}. "
                                "Giải nén ktvic_processed_d796bc1.zip ở gốc repo hoặc truyền --processed-dir.")
    return path


def resolve_image_root(root="."):
    """Folder that contains train-images/ and public-test-images/ (manifest paths are relative to it)."""
    path = Path(root) / "data/raw/ktvic"
    if not ((path / "train-images").is_dir() and (path / "public-test-images").is_dir()):
        raise FileNotFoundError(f"Không thấy train-images/ và public-test-images/ trong {path}. "
                                "Giải nén hai gói ảnh theo README hoặc truyền --image-root.")
    return path
