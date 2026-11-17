"""Score predicted Vietnamese captions (BLEU-1..4, CIDEr) against all references of each image.

Example:
  python -m scripts.score_captions --predictions runs/<run>/val_predictions.json --split val
"""

import argparse
import json
import sys
from pathlib import Path

from src.data.paths import resolve_image_root, resolve_processed_dir
from src.evaluation.caption_scorer import (
    ScorerInputError, load_image_paths, load_predictions, load_references, score_captions,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", type=Path, required=True,
                        help='JSON/JSONL: [{"image_id": 3, "caption": "..."}] or {"3": "..."}')
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--processed-dir", type=Path, help="Folder with *_manifest.jsonl (auto-detected)")
    parser.add_argument("--image-root", type=Path, help="Check that each scored image file exists here (auto-detected)")
    parser.add_argument("--no-image-check", action="store_true", help="Skip checking image files")
    parser.add_argument("--output", type=Path, help="Also write the result as JSON")
    args = parser.parse_args(argv)

    try:
        processed = args.processed_dir or resolve_processed_dir()
        manifest = processed / f"{args.split}_manifest.jsonl"
        references = load_references(manifest)
        options = {}
        if not args.no_image_check:
            options.update(image_paths=load_image_paths(manifest),
                           image_root=args.image_root or resolve_image_root())
        result = score_captions(load_predictions(args.predictions), references, **options)
    except (ScorerInputError, FileNotFoundError) as error:
        print(f"LỖI CHẤM ĐIỂM ({args.split}):\n{error}", file=sys.stderr)
        return 2

    result["split"] = args.split
    print(f"Tập {args.split}: {result['num_images']} ảnh, {result['num_references']} câu tham chiếu "
          f"(thang: {result['scale']})")
    for name, value in result["scores"].items():
        print(f"  {name:<7} {value:7.2f}")
    print("  thư viện:", ", ".join(f"{k} {v}" for k, v in result["libraries"].items()))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    sys.exit(main())
