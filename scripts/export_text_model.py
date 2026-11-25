"""Copy the pinned MiniLM text model from the Hugging Face cache into one project folder.

The folder can be carried to a machine with no network and no Hugging Face cache; the demo reads it directly:
  python -m scripts.export_text_model
"""

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

from src.retrieval.visual_search import (DEFAULT_TEXT_MODEL_DIR, SOURCE_FILE, TEXT_MODEL, TEXT_MODEL_FILES,
                                         TEXT_REVISION, check_text_model_dir)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_TEXT_MODEL_DIR)
    args = parser.parse_args()
    try:
        from huggingface_hub import snapshot_download
        try:
            cached = Path(snapshot_download(TEXT_MODEL, revision=TEXT_REVISION, local_files_only=True,
                                            allow_patterns=list(TEXT_MODEL_FILES)))
        except Exception as error:  # LocalEntryNotFoundError and friends: the cache has no such snapshot
            raise FileNotFoundError(f"Cache Hugging Face chưa có {TEXT_MODEL} ({TEXT_REVISION}). Tải một lần trên máy "
                                    "có mạng theo README rồi chạy lại.") from error
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for name in TEXT_MODEL_FILES:
            shutil.copyfile(cached / name, args.output_dir / name)
        files = {name: sha256(args.output_dir / name) for name in TEXT_MODEL_FILES}
        (args.output_dir / SOURCE_FILE).write_text(
            json.dumps({"model": TEXT_MODEL, "revision": TEXT_REVISION, "sha256": files}, indent=2), encoding="utf-8")
        check_text_model_dir(args.output_dir)
    except (OSError, ValueError) as error:
        sys.exit(f"Không xuất được mô hình văn bản: {error}")
    size = sum((args.output_dir / name).stat().st_size for name in TEXT_MODEL_FILES) / 1e6
    print(f"Đã xuất {len(files)} tệp ({size:.0f} MB) vào {args.output_dir}. Chép thư mục này sang máy chạy offline.")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    main()
