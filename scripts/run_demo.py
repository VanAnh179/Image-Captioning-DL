"""Open the offline demo window: pick an image to caption it, or type a Vietnamese sentence to find 5 images.

  python -m scripts.run_demo
Runs from local files only (A checkpoint, TF-IDF index of A's captions, KTVIC images, cached ResNet-50 weights).
"""

import argparse
import sys
import tkinter as tk
from pathlib import Path

import torch

from src.demo.app import DemoApp
from src.demo.engine import DEFAULT_BEAM_SIZE, DEFAULT_CHECKPOINTS, DEFAULT_INDEX, DemoEngine, first_existing


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, help="A's best.pt (default: the version-list path, else ./best.pt)")
    parser.add_argument("--index", type=Path, default=Path(DEFAULT_INDEX))
    parser.add_argument("--image-root", type=Path, default=Path("data/raw/ktvic"))
    parser.add_argument("--beam-size", type=int, default=DEFAULT_BEAM_SIZE, help="beam width for captions (1 = greedy)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    try:
        checkpoint = args.checkpoint or first_existing(DEFAULT_CHECKPOINTS)
        print(f"Đang nạp mô hình A ({checkpoint}) và chỉ mục ({args.index}) trên {args.device}…", flush=True)
        engine = DemoEngine(checkpoint, args.index, args.image_root, args.device, beam_size=args.beam_size)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        sys.exit(f"Không mở được demo: {error}")
    try:  # sharp text on high-DPI Windows screens
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except (AttributeError, OSError):
        pass
    root = tk.Tk()
    DemoApp(root, engine, start_dir=str(args.image_root))
    root.mainloop()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    main()
