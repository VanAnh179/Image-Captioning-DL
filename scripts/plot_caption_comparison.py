"""Render validation images with A/A0 captions selected by the comparison script."""

import argparse
import json
import math
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageOps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=Path("data/raw/ktvic"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.comparison.read_text(encoding="utf-8"))
    examples = report["examples"][:6]
    if not examples:
        print("A and A0 produced identical captions; there are no changed examples to plot.")
        return
    rows = math.ceil(len(examples) / 2)
    fig = plt.figure(figsize=(16, rows * 6.5), layout="constrained")
    grid = fig.add_gridspec(rows, 2)
    for position, example in enumerate(examples):
        cell = grid[position // 2, position % 2].subgridspec(2, 1, height_ratios=[3, 2.3])
        image_ax = fig.add_subplot(cell[0])
        text_ax = fig.add_subplot(cell[1])
        with Image.open(args.image_root / example["image_path"]) as source:
            image_ax.imshow(ImageOps.exif_transpose(source).convert("RGB"))
        image_ax.set_title(f"ID {example['image_id']} | CIDEr A - A0 = {example['delta_a_minus_a0']:+.2f}")
        image_ax.axis("off")
        text = "\n\n".join(textwrap.fill(label + value, width=78) for label, value in (
            ("A: ", example["caption_a"]), ("A0: ", example["caption_a0"]),
            ("Tham chiếu: ", example["references"][0])))
        text_ax.text(0, 1, text, va="top", fontsize=10)
        text_ax.axis("off")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
