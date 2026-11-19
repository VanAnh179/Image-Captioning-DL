"""Run a Vietnamese caption query on CPU and return ranked image IDs and paths."""

import argparse
import json
import sys
from pathlib import Path

from src.retrieval.caption_search import CaptionSearchIndex, SearchInputError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=Path("artifacts/retrieval/ktvic_a_val.npz"))
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--image-root", type=Path, default=Path("data/raw/ktvic"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    index = CaptionSearchIndex.load(args.index)
    try:
        results = index.search(args.query, args.top_k, args.image_root)
    except SearchInputError as error:
        parser.error(str(error))
    text = json.dumps(results, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    main()
