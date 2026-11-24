"""Open the offline demo as a local web page: upload an image to get a caption, type a sentence to find 5 images.

  python -m scripts.run_web_demo
Then open http://127.0.0.1:8000. Runs from local files only, same as scripts.run_demo.
"""

import argparse
import os
import sys
import webbrowser
from pathlib import Path

# Windows conda installs often load two copies of the OpenMP runtime (torch + numpy/MKL) and abort with OMP Error #15.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch

from src.demo.engine import DEFAULT_CHECKPOINTS, DEFAULT_INDEX, DemoEngine, first_existing
from src.demo.web import serve


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, help="A's best.pt (default: the version-list path, else ./best.pt)")
    parser.add_argument("--index", type=Path, default=Path(DEFAULT_INDEX))
    parser.add_argument("--image-root", type=Path, default=Path("data/raw/ktvic"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true", help="do not open the page automatically")
    args = parser.parse_args()
    try:
        checkpoint = args.checkpoint or first_existing(DEFAULT_CHECKPOINTS)
        print(f"Đang nạp mô hình A ({checkpoint}) và chỉ mục ({args.index}) trên {args.device}…", flush=True)
        engine = DemoEngine(checkpoint, args.index, args.image_root, args.device)
        server = serve(engine, port=args.port)
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as error:
        sys.exit(f"Không mở được demo: {error}")
    url = f"http://127.0.0.1:{args.port}"
    print(f"Demo đang chạy tại {url} (chỉ máy này truy cập được). Nhấn Ctrl+C để dừng.", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Đã dừng.")
    finally:
        server.server_close()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    main()
