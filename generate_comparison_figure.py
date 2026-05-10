"""
Generate paper-style comparison figures from qualitative_results.

Outputs:
    Paper_Comparison_Visuals/comparison_figure_x2.png
    Paper_Comparison_Visuals/comparison_figure_x4.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

matplotlib.rcParams["font.family"] = "DejaVu Sans"

BASE_DIR = Path(__file__).resolve().parent
QUAL_ROOT = BASE_DIR / "qualitative_results"
OUTPUT_DIR = BASE_DIR / "Paper_Comparison_Visuals"

MODEL_SPECS = [
    ("LR", "LR Input", "lr"),
    ("Bicubic", "Bicubic", "bicubic"),
    ("EDSR-MMHCA", "EDSR-MMHCA", "edsr"),
    ("Swin-MMHCA", "Swin-MMHCA", "swin"),
    ("GT", "Ground Truth", "gt"),
]

COLORS = {
    "lr": "#444444",
    "bicubic": "#E59400",
    "edsr": "#1B7F3E",
    "swin": "#1A3FBF",
    "gt": "#8B1EA4",
}

TARGET_SIZE = (342, 341)  # width, height
ZOOM_CENTER_FRAC = (120 / 256.0, 140 / 256.0)  # (x, y)
ZOOM_SIZE_FRAC = 40 / 256.0
DPI = 220


def extract_sample_name(filename: str) -> str:
    parts = filename.split("_")
    if len(parts) >= 2 and parts[0] == "sample":
        return f"{parts[0]}_{parts[1]}"
    return Path(filename).stem


def list_sample_names(scale_dir: Path):
    common = None
    for folder, _, _ in MODEL_SPECS:
        names = {extract_sample_name(path.name) for path in (scale_dir / folder).glob("*.png")}
        common = names if common is None else common & names
    return sorted(common or [])


def find_sample_path(scale_dir: Path, folder: str, sample_name: str) -> Path:
    matches = sorted((scale_dir / folder).glob(f"{sample_name}_*.png"))
    if not matches:
        raise FileNotFoundError(f"Missing {sample_name} in {(scale_dir / folder)}")
    return matches[0]


def load_gray(path: Path) -> Image.Image:
    return Image.open(path).convert("L")


def fit_to_canvas(img: Image.Image, target_size):
    target_w, target_h = target_size
    src_w, src_h = img.size
    scale = min(target_w / src_w, target_h / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resized = img.resize((new_w, new_h), Image.Resampling.BICUBIC)

    canvas = Image.new("L", (target_w, target_h), color=255)
    x_off = (target_w - new_w) // 2
    y_off = (target_h - new_h) // 2
    canvas.paste(resized, (x_off, y_off))
    meta = {"x_off": x_off, "y_off": y_off, "width": new_w, "height": new_h}
    return canvas, meta


def zoom_box_from_meta(meta):
    x_off = meta["x_off"]
    y_off = meta["y_off"]
    disp_w = meta["width"]
    disp_h = meta["height"]

    cx = x_off + ZOOM_CENTER_FRAC[0] * disp_w
    cy = y_off + ZOOM_CENTER_FRAC[1] * disp_h
    box_w = max(10, int(round(ZOOM_SIZE_FRAC * disp_w)))
    box_h = max(10, int(round(ZOOM_SIZE_FRAC * disp_h)))

    left = int(round(cx - box_w / 2))
    top = int(round(cy - box_h / 2))
    right = left + box_w
    bottom = top + box_h

    target_w, target_h = TARGET_SIZE
    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > target_w:
        left -= right - target_w
        right = target_w
    if bottom > target_h:
        top -= bottom - target_h
        bottom = target_h

    return left, top, right, bottom


def crop_patch(img: Image.Image, box):
    patch = img.crop(box)
    return patch.resize(TARGET_SIZE, Image.Resampling.NEAREST)


def show_img(ax, img, border_color=None, border_lw=2.0):
    ax.imshow(np.asarray(img), cmap="gray", vmin=0, vmax=255, aspect="equal", interpolation="nearest")
    ax.axis("off")
    if border_color:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(border_color)
            spine.set_linewidth(border_lw)


def draw_box(ax, box, color, lw=1.5):
    left, top, right, bottom = box
    rect = mpatches.Rectangle(
        (left, top),
        right - left,
        bottom - top,
        linewidth=lw,
        edgecolor=color,
        facecolor="none",
    )
    ax.add_patch(rect)


def render_scale(scale: int):
    scale_dir = QUAL_ROOT / f"x{scale}"
    sample_names = list_sample_names(scale_dir)
    if not sample_names:
        raise RuntimeError(f"No shared samples found in {scale_dir}")

    n_rows = len(sample_names) * 2
    n_cols = len(MODEL_SPECS)
    row_heights = []
    for _ in sample_names:
        row_heights.extend([2.5, 1.7])

    fig_w = n_cols * 2.2 + 0.6
    fig_h = sum(row_heights) + 0.6
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=DPI)
    fig.patch.set_facecolor("white")

    gs = fig.add_gridspec(
        n_rows,
        n_cols,
        height_ratios=row_heights,
        hspace=0.04,
        wspace=0.03,
        top=0.955,
        bottom=0.02,
        left=0.06,
        right=0.995,
    )

    for col_idx, (_, label, color_key) in enumerate(MODEL_SPECS):
        ax_h = fig.add_axes([0.06 + col_idx * (0.935 / n_cols), 0.963, 0.935 / n_cols, 0.025])
        ax_h.axis("off")
        ax_h.text(
            0.5,
            0.5,
            label,
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            color=COLORS[color_key],
            transform=ax_h.transAxes,
        )

    for sample_idx, sample_name in enumerate(sample_names):
        row_full = sample_idx * 2
        row_patch = row_full + 1

        for col_idx, (folder, _, color_key) in enumerate(MODEL_SPECS):
            img_path = find_sample_path(scale_dir, folder, sample_name)
            img = load_gray(img_path)
            aligned, meta = fit_to_canvas(img, TARGET_SIZE)
            box = zoom_box_from_meta(meta)
            patch = crop_patch(aligned, box)

            ax_full = fig.add_subplot(gs[row_full, col_idx])
            show_img(ax_full, aligned)
            draw_box(ax_full, box, COLORS[color_key], lw=1.4)

            ax_patch = fig.add_subplot(gs[row_patch, col_idx])
            show_img(ax_patch, patch, border_color=COLORS[color_key], border_lw=2.0)

            if col_idx == 0:
                ax_full.set_ylabel(
                    sample_name.replace("_", " "),
                    fontsize=7,
                    rotation=90,
                    va="center",
                    labelpad=6,
                )

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / f"comparison_figure_x{scale}.png"
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    render_scale(4)
    render_scale(2)
