"""One place for the project folders, all relative to the repo root (Git-ignored, see README).

  data/raw/ktvic              KTVIC annotation JSON and the two image packs
  data/processed/ktvic        manifests, ID lists and the vocabulary made by prepare_ktvic_dataset
  artifacts/features/ktvic    cached ResNet-50 and CLIP region features
  artifacts/checkpoints/ktvic one folder per trained run, e.g. A/best.pt
  artifacts/retrieval         TF-IDF search indexes
  logs                        console logs of long commands
  reports                     score tables, predictions and handoff files
"""

from pathlib import Path

RAW_DIR = Path("data/raw/ktvic")
PROCESSED_DIR = Path("data/processed/ktvic")
FEATURE_DIR = Path("artifacts/features/ktvic")
CHECKPOINT_DIR = Path("artifacts/checkpoints/ktvic")
RETRIEVAL_DIR = Path("artifacts/retrieval")
LOG_DIR = Path("logs")
REPORT_DIR = Path("reports")


def resolve_processed_dir(root="."):
    """Folder holding {train,val,test}_ids.json, manifests and the vocabulary."""
    path = Path(root) / PROCESSED_DIR
    if not (path / "train_manifest.jsonl").is_file():
        raise FileNotFoundError(f"Không thấy train_manifest.jsonl trong {path}. "
                                "Chạy python -m scripts.prepare_ktvic_dataset hoặc truyền --processed-dir.")
    return path


def resolve_image_root(root="."):
    """Folder that contains train-images/ and public-test-images/ (manifest paths are relative to it)."""
    path = Path(root) / RAW_DIR
    if not ((path / "train-images").is_dir() and (path / "public-test-images").is_dir()):
        raise FileNotFoundError(f"Không thấy train-images/ và public-test-images/ trong {path}. "
                                "Giải nén hai gói ảnh theo README hoặc truyền --image-root.")
    return path
