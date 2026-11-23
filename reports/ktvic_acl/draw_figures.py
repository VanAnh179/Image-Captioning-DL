"""Rebuild the validation plot and test illustrations from the local handoff."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import MultipleLocator
import numpy as np
from PIL import Image, ImageOps


PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parents[1]
FIGURES = PAPER / "figures"
ATTENTION = (
    ROOT / "artifacts/handoff/Phieu_25_anh_va_attention_KTVIC"
    / "reports/caption_test_attention/ban_do_theo_tu.json"
)
TEST_IMAGES = ROOT / "data/raw/ktvic/public-test-images"
SEED_RESULTS = ROOT / "reports/clip_transformer/nam_seed_B_C_kaggle.json"
IMAGE_SIZE = 420


def plot_seeds():
    data = json.loads(SEED_RESULTS.read_text(encoding="utf-8"))
    by_model = {
        model: {run["seed"]: run["CIDEr"] for run in data[model]["runs"]}
        for model in ("B", "C")
    }

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.28, 2.02), layout="constrained")
    seeds = data["seeds"]
    rows = range(len(seeds))
    for row, seed in zip(rows, seeds):
        start, end = (by_model[model][seed] for model in ("B", "C"))
        ax.hlines(row, start, end, color="#9aaebd", linewidth=1.6, zorder=2)
    for model, color, marker, name in (
        ("B", "#24496d", "o", "B · ResNet-50"),
        ("C", "#9c4717", "D", "C · CLIP"),
    ):
        ax.scatter(
            [by_model[model][seed] for seed in seeds], rows, color=color,
            marker=marker, s=28, linewidths=0, label=name, zorder=3,
        )
    ax.invert_yaxis()
    ax.set_xlim(68, 85)
    ax.set_yticks(list(rows), seeds)
    ax.set_ylabel("Seed")
    ax.set_xlabel("CIDEr-D")
    ax.xaxis.set_major_locator(MultipleLocator(4))
    ax.grid(axis="x", color="#dce3e9", linewidth=.7)
    ax.spines[["top", "left", "right"]].set_visible(False)
    ax.spines["bottom"].set_color("#a9b6c0")
    ax.tick_params(length=0)
    ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.01), ncol=2, frameon=False)
    fig.savefig(FIGURES / "seed_comparison.pdf", metadata={"CreationDate": None})
    plt.close(fig)


def load_test_image(image_id):
    with Image.open(TEST_IMAGES / f"{image_id:011d}.jpg") as raw:
        return ImageOps.exif_transpose(raw).convert("RGB")


def draw_test_examples():
    for image_id in (297, 4568):
        image = ImageOps.pad(
            load_test_image(image_id), (IMAGE_SIZE, IMAGE_SIZE),
            Image.Resampling.LANCZOS, color="#f0f2f4",
        )
        image.save(FIGURES / f"scene_{image_id}.jpg", quality=93, optimize=True)

    # The ResNet-50 preprocessing uses a square resize, so its grid covers
    # the full displayed image. Keep the same geometry for the two panels.
    scene = load_test_image(3091).resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    scene.save(FIGURES / "scene_3091.jpg", quality=93, optimize=True)
    data = json.loads(ATTENTION.read_text(encoding="utf-8"))
    case = next(row for row in data["cases"] if row["image_id"] == 3091)
    token = next(row for row in case["tokens"] if row["token"] == "chụp")
    weights = np.array(token["models"]["A"]["weights"])
    max_row, max_col = np.unravel_index(weights.argmax(), weights.shape)

    fig, ax = plt.subplots(figsize=(2, 2), dpi=IMAGE_SIZE // 2)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.imshow(scene, extent=(0, 7, 7, 0))
    ax.imshow(
        weights, extent=(0, 7, 7, 0), cmap="magma", interpolation="nearest",
        vmin=0, vmax=weights.max(), alpha=.6 * weights / weights.max(),
    )
    ax.vlines(range(1, 7), 0, 7, color="white", alpha=.55, lw=.35)
    ax.hlines(range(1, 7), 0, 7, color="white", alpha=.55, lw=.35)
    ax.add_patch(Rectangle((max_col, max_row), 1, 1, fill=False, edgecolor="white", lw=1.8))
    ax.axis("off")
    fig.savefig(FIGURES / "attention_3091.jpg", pil_kwargs={"quality": 93, "optimize": True})
    plt.close(fig)


if __name__ == "__main__":
    FIGURES.mkdir(exist_ok=True)
    plot_seeds()
    draw_test_examples()
