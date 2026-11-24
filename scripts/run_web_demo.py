"""Open the offline demo as a local web page: upload an image to get a caption, type a sentence to find 5 images.

  python -m scripts.run_web_demo
Then open http://127.0.0.1:8000. Runs from local files only, same as scripts.run_demo.
"""

import argparse
import os
import subprocess
import sys
import tempfile
import webbrowser
from pathlib import Path

import torch

from src.demo.engine import DEFAULT_CHECKPOINTS, DEFAULT_INDEX, DemoEngine, first_existing
from src.demo.web import serve


def open_app_window(url):
    """Open url as a standalone window with Edge or Chrome app mode; False when neither is installed."""
    roots = [os.environ.get(name) for name in ("ProgramFiles(x86)", "ProgramFiles", "LOCALAPPDATA")]
    candidates = [Path(root) / relative for root in roots if root for relative in
                  ("Microsoft/Edge/Application/msedge.exe", "Google/Chrome/Application/chrome.exe")]
    for browser in candidates:
        if browser.is_file():
            # A separate profile folder makes the browser start its own process, so the app window is not folded into
            # the user's open browser as another tab.
            profile = Path(tempfile.gettempdir()) / "vicaption_app_profile"
            try:
                subprocess.Popen([str(browser), f"--app={url}", "--window-size=1200,900", f"--user-data-dir={profile}",
                                  "--no-first-run", "--no-default-browser-check"])
            except OSError:
                continue
            return True
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, help="A's best.pt (default: the version-list path, else ./best.pt)")
    parser.add_argument("--index", type=Path, default=Path(DEFAULT_INDEX))
    parser.add_argument("--image-root", type=Path, default=Path("data/raw/ktvic"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true", help="do not open the page automatically")
    parser.add_argument("--window", action="store_true",
                        help="open the page in its own app-style window (Edge or Chrome, no tabs or address bar)")
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error("--port phải nằm trong khoảng 0 đến 65535.")
    try:
        checkpoint = args.checkpoint or first_existing(DEFAULT_CHECKPOINTS)
        print(f"Đang nạp mô hình A ({checkpoint}) và chỉ mục ({args.index}) trên {args.device}…", flush=True)
        engine = DemoEngine(checkpoint, args.index, args.image_root, args.device)
        server = serve(engine, port=args.port)
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as error:
        sys.exit(f"Không mở được demo: {error}")
    url = f"http://127.0.0.1:{server.server_port}"
    print(f"Demo đang chạy tại {url} (chỉ máy này truy cập được). Nhấn Ctrl+C để dừng.", flush=True)
    if not args.no_browser:
        if not (args.window and open_app_window(url)):
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
