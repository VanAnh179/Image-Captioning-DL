"""Check that this machine and folder are ready: packages, GPU, ResNet-50 weights and the project folders.

  python -m scripts.check_setup
Exit code 1 means a package is missing. Everything else is a warning with the command that fixes it.
"""

import argparse
import importlib.metadata
import sys
from pathlib import Path

from src.data.paths import CHECKPOINT_DIR, FEATURE_DIR, LOG_DIR, PROCESSED_DIR, RAW_DIR, REPORT_DIR, RETRIEVAL_DIR

RESNET_WEIGHTS = "resnet50-0676ba61.pth"
DISTRIBUTIONS = {"PyYAML": "yaml"}  # pip name -> import name, only where they differ


def read_requirements(path):
    """{pip name: pinned version} from the == lines of requirements.txt."""
    pins = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if "==" in line:
            name, _, version = line.partition("==")
            pins[name.strip()] = version.strip()
    return pins


def package_status(pins):
    """[(name, wanted, installed or None, state)] with state ok, khác phiên bản or thiếu."""
    rows = []
    for name, wanted in pins.items():
        try:
            installed = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            rows.append((name, wanted, None, "thiếu"))
            continue
        base = installed.split("+")[0]
        rows.append((name, wanted, installed, "ok" if base == wanted else "khác phiên bản"))
    return rows


def folder_summary(root="."):
    """[(label, path, found, note)] for the folders every command reads or writes."""
    root = Path(root)

    def count(folder, pattern):
        return len(list((root / folder).glob(pattern))) if (root / folder).is_dir() else 0

    raw = root / RAW_DIR
    rows = [
        ("Dữ liệu thô", RAW_DIR, all((raw / name).exists() for name in ("train_data.json", "test_data.json", "train-images", "public-test-images")),
         "cần train_data.json, test_data.json, train-images/, public-test-images/"),
        ("Dữ liệu đã xử lý", PROCESSED_DIR, (root / PROCESSED_DIR / "train_manifest.jsonl").is_file(),
         "chạy python -m scripts.prepare_ktvic_dataset --seed 42"),
        ("Đặc trưng ảnh", FEATURE_DIR, count(FEATURE_DIR, "resnet50_7x7_*.pt") == 3,
         "chạy python -m scripts.extract_region_features (cache CLIP: scripts.extract_clip_features)"),
        ("Checkpoint", CHECKPOINT_DIR, count(CHECKPOINT_DIR, "*/best.pt") > 0,
         "đặt mỗi lần chạy ở " + str(CHECKPOINT_DIR / "<tên>" / "best.pt")),
        ("Chỉ mục tìm ảnh", RETRIEVAL_DIR, count(RETRIEVAL_DIR, "*.npz") > 0,
         "chạy python -m scripts.build_image_search_index"),
        ("Log", LOG_DIR, (root / LOG_DIR).is_dir(), "tự tạo khi chạy lệnh huấn luyện hoặc trích đặc trưng"),
        ("Kết quả và bảng", REPORT_DIR, (root / REPORT_DIR).is_dir(), "tự tạo khi tái tạo bảng điểm"),
    ]
    return rows


def resnet_weights_path():
    import torch

    return Path(torch.hub.get_dir()) / "checkpoints" / RESNET_WEIGHTS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requirements", type=Path, default=Path("requirements.txt"))
    args = parser.parse_args()
    missing = 0

    print(f"Python {sys.version.split()[0]}")
    print("\nGói (so với requirements.txt):")
    for name, wanted, installed, state in package_status(read_requirements(args.requirements)):
        print(f"  {name:<14} cần {wanted:<8} có {installed or '-':<14} {state}")
        missing += state == "thiếu"
    try:
        import tkinter  # noqa: F401
        print("  tkinter        có (chỉ cần cho bản demo cửa sổ scripts.run_demo)")
    except ImportError:
        print("  tkinter        thiếu (chỉ cần cho scripts.run_demo; trang web scripts.run_web_demo vẫn chạy)")

    print("\nGPU và trọng số:")
    try:
        import torch

        if torch.cuda.is_available():
            print(f"  CUDA: {torch.cuda.get_device_name(0)} (torch {torch.__version__})")
        else:
            print(f"  Không có CUDA (torch {torch.__version__}): thêm --device cpu khi chạy lệnh. Muốn dùng GPU, xem mục Môi trường trong README.")
        weights = resnet_weights_path()
        print(f"  ResNet-50 {RESNET_WEIGHTS}: " + ("có" if weights.is_file() else
              f"chưa có trong {weights.parent}. Chạy một lần khi có mạng: python -c \"import torchvision; torchvision.models.resnet50(weights='IMAGENET1K_V1')\""))
    except ImportError:
        print("  Chưa cài torch.")

    print("\nThư mục dự án (đường dẫn tính từ gốc repo):")
    for label, path, found, note in folder_summary():
        print(f"  [{'có' if found else 'chưa'}] {label:<18} {str(path):<30} " + ("" if found else note))
    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
