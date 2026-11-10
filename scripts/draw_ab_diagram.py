"""Draw the image -> regions -> words diagram for model A (LSTM + soft attention) and model B (Transformer)."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

FILL = {"shared": "#dbe4ee", "own": "#f2f2f2", "A": "#fde3c8", "B": "#cfeae6"}
EDGE = {"shared": "#46607a", "own": "#666666", "A": "#b8641a", "B": "#1f7a6e"}


def box(ax, x, y, w, h, text, kind, size=9.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=FILL[kind], ec=EDGE[kind], lw=1.4))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size,
            fontweight="bold" if bold else "normal", linespacing=1.35)
    return (x, y, w, h)


def arrow(ax, start, end, color="#333333", rad=0.0, lw=1.5, ls="-"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13, color=color, lw=lw,
                                 linestyle=ls, connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


def right(b):
    return (b[0] + b[2], b[1] + b[3] / 2)


def left(b):
    return (b[0], b[1] + b[3] / 2)


def top(b, f=0.5):
    return (b[0] + b[2] * f, b[1] + b[3])


def bottom(b, f=0.5):
    return (b[0] + b[2] * f, b[1])


def draw(path):
    fig, ax = plt.subplots(figsize=(17.6, 12.4))
    ax.set_xlim(0, 17.8)
    ax.set_ylim(0, 12.6)
    ax.axis("off")
    ax.text(0.2, 12.3, "Đường đi từ ảnh → vùng đặc trưng → từng từ: mô hình A (LSTM + soft attention) và B (Transformer decoder)",
            fontsize=13.5, fontweight="bold", va="center")

    # --- Shared path (identical for A and B)
    ax.text(0.2, 11.75, "DÙNG CHUNG CHO A VÀ B: ảnh → 49 vùng  (tính một lần, lưu cache theo image_id)", fontsize=10.5,
            color=EDGE["shared"], fontweight="bold", va="center")
    y, h = 10.0, 1.35
    img = box(ax, 0.2, y, 2.3, h, "Ảnh RGB\n224×224×3\n(resize bilinear)", "shared")
    cnn = box(ax, 3.1, y, 2.9, h, "ResNet-50 ĐÓNG BĂNG\nIMAGENET1K_V1, eval()\nconv1 … layer4 (bỏ avgpool, fc)", "shared")
    grid = box(ax, 6.6, y, 2.6, h, "Bản đồ đặc trưng\n2048 × 7 × 7\n(stride 32: mỗi ô ≈ 32×32 px)", "shared")
    reg = box(ax, 9.8, y, 3.0, h, "49 vùng  a₁ … a₄₉\ntensor [49, 2048]\nvùng i = hàng×7 + cột", "shared", bold=True)
    mem = box(ax, 13.4, y, 3.4, h, "Chiếu vùng + embedding vị trí\ncùng kiến trúc, trọng số riêng\ncho A và B  ⇒  M [49, d]", "own", size=9)
    ax.text(15.1, 11.55, "TỪNG MÔ HÌNH TỰ HUẤN LUYỆN", fontsize=9, color=EDGE["own"], fontweight="bold", ha="center")
    for a, b in ((img, cnn), (cnn, grid), (grid, reg), (reg, mem)):
        arrow(ax, right(a), left(b))
    ax.text(11.3, y - 0.27, "cache: resnet50_7x7_{split}.pt  (image_id → [49, 2048], fp16)", fontsize=8.5,
            color="#555555", ha="center")

    # --- M bus between the two lanes: both models read the same tensor
    box(ax, 0.3, 4.5, 16.5, 0.8,
        "M  [49, d]  —  mỗi mô hình tự chiếu cùng 49 vùng của image_id thành M của mình "
        "(A: đối tượng được chú ý;  B: K, V của cross-attention)", "own", size=9.5, bold=True)
    ax.plot([16.8, 17.45, 17.45], [10.675, 10.675, 4.9], color=EDGE["shared"], lw=1.8, solid_capstyle="round")
    arrow(ax, (17.45, 4.9), (16.8, 4.9), color=EDGE["shared"], lw=1.8)

    # --- Lane A
    ax.add_patch(FancyBboxPatch((0.1, 5.7), 16.8, 3.65, boxstyle="round,pad=0.02,rounding_size=0.2",
                                fc="none", ec=EDGE["A"], lw=1.2, ls="--"))
    ax.text(0.3, 9.1, "A — LSTM + soft attention (Show, Attend and Tell): mỗi bước t nhìn lại 49 vùng bằng một phân phối αₜ",
            fontsize=10.5, color=EDGE["A"], fontweight="bold", va="center")
    ya, ha = 6.2, 1.45
    a1 = box(ax, 0.3, ya, 3.0, ha, "Điểm chú ý (additive)\neₜᵢ = wᵀ tanh(W_m·mᵢ + W_h·hₜ₋₁)\ncho từng vùng i = 1..49", "A", size=9)
    a2 = box(ax, 3.9, ya, 2.4, ha, "αₜ = softmax(eₜ)\ntrên 49 vùng\nΣᵢ αₜᵢ = 1", "A", bold=True, size=9.5)
    a3 = box(ax, 6.9, ya, 2.6, ha, "Vector ngữ cảnh\nẑₜ = Σᵢ αₜᵢ · mᵢ\n[d]  (tuỳ chọn cổng β)", "A", size=9.5)
    a4 = box(ax, 10.1, ya, 3.1, ha, "LSTMCell\ninput = [E(yₜ₋₁) ; ẑₜ]\n(hₜ₋₁, cₜ₋₁) → (hₜ, cₜ)", "A", size=9.5)
    a5 = box(ax, 13.8, ya, 2.9, ha, "Linear(hₜ) → softmax\ntrên 1.716 token\n⇒ từ wₜ", "A", bold=True, size=9.5)
    for p, q in ((a1, a2), (a2, a3), (a3, a4), (a4, a5)):
        arrow(ax, right(p), left(q))
    arrow(ax, (1.8, 5.3), bottom(a1, 0.5), color=EDGE["shared"], lw=1.8)
    arrow(ax, (8.2, 5.3), bottom(a3, 0.5), color=EDGE["shared"], lw=1.8)
    ax.text(2.0, 5.52, "M", fontsize=9, color=EDGE["shared"], fontweight="bold")
    ax.text(8.4, 5.52, "M", fontsize=9, color=EDGE["shared"], fontweight="bold")
    arrow(ax, top(a4, 0.5), top(a1, 0.5), color=EDGE["A"], rad=0.11, ls=(0, (4, 3)), lw=1.3)
    ax.text(7.0, 8.42, "hₜ₋₁ quay lại làm truy vấn cho bước sau (tuần tự: bước t cần bước t−1, cả khi huấn luyện)",
            fontsize=8.8, color=EDGE["A"], ha="center")
    ax.text(10.1, 5.95, "Khởi tạo h₀, c₀ = MLP(trung bình 49 vùng).\nTuỳ chọn: doubly-stochastic Σₜ αₜᵢ ≈ 1.",
            fontsize=8.5, color="#444444", va="center", linespacing=1.3)

    # --- Lane B
    ax.add_patch(FancyBboxPatch((0.1, 0.95), 16.8, 3.25, boxstyle="round,pad=0.02,rounding_size=0.2",
                                fc="none", ec=EDGE["B"], lw=1.2, ls="--"))
    ax.text(0.3, 3.8, "B — Transformer decoder (Attention Is All You Need):\nmọi từ đã sinh cùng nhìn 49 vùng qua cross-attention",
            fontsize=10.5, color=EDGE["B"], fontweight="bold", va="center", linespacing=1.4)
    yb, hb = 1.5, 1.5
    b1 = box(ax, 0.3, yb, 2.6, hb, "Từ đã sinh y₁..yₜ₋₁\nEmbedding + mã hoá\nvị trí từ (sin/cos)", "B", size=9.2)
    b2 = box(ax, 3.4, yb, 2.8, hb, "Masked self-attention\n(mặt nạ nhân quả)\ntừ chỉ thấy các từ trước", "B", size=9.2)
    b3 = box(ax, 6.7, yb, 3.9, hb, "CROSS-attention\nQ từ trạng thái từ;  K, V từ M\nsoftmax(Q·Kᵀ/√d)·V,  H đầu", "B", bold=True, size=9.2)
    b4 = box(ax, 11.1, yb, 2.5, hb, "FFN + residual\n+ LayerNorm\n(lặp L lớp)", "B", size=9.2)
    b5 = box(ax, 14.0, yb, 2.8, hb, "Linear → softmax\ntrên 1.716 token\n⇒ từ wₜ", "B", bold=True, size=9.2)
    for p, q in ((b1, b2), (b2, b3), (b3, b4), (b4, b5)):
        arrow(ax, right(p), left(q))
    arrow(ax, (9.6, 4.5), top(b3, 0.745), color=EDGE["shared"], lw=1.8)
    ax.text(9.8, 3.55, "K, V = M", fontsize=9, color=EDGE["shared"], fontweight="bold")

    ax.text(0.3, 0.4, "Dùng chung ở cả hai nhánh: từ vựng 1.716 token (PAD/BOS/EOS/UNK), BOS→nhãn→EOS + mask PAD, cross-entropy có mask, "
            "bộ chấm BLEU-1..4 / CIDEr trên cả năm câu tham chiếu.", fontsize=9, color="#333333", va="center")
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    if str(path).endswith(".png"):
        fig.savefig(Path(path).with_suffix(".svg"), bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("docs/so_do_AB.png"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    draw(args.output)
    print("saved", args.output)
