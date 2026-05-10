import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


BASE_DIR = Path(__file__).resolve().parent
QUAL_ROOT = BASE_DIR / "qualitative_results"
OUTPUT_ROOT = BASE_DIR / "Paper_Comparison_Visuals"

MODEL_ORDER = [
    ("LR", "LR Input"),
    ("Bicubic", "Bicubic"),
    ("EDSR-MMHCA", "EDSR-MMHCA"),
    ("Swin-MMHCA", "Swin-MMHCA (Ours)"),
    ("GT", "Ground Truth"),
]

# Reuse the paper-style zoom region, normalized from the 256x256 reference view.
ZOOM_CENTER_FRAC = (120 / 256.0, 140 / 256.0)  # (x, y)
ZOOM_SIZE_FRAC = 40 / 256.0
TARGET_SIZE = (342, 341)  # width, height


def list_sample_names(scale_dir: Path):
    sample_names = None
    for folder, _ in MODEL_ORDER:
        folder_dir = scale_dir / folder
        files = sorted(folder_dir.glob("*.png"))
        stems = {extract_sample_name(path.name) for path in files}
        sample_names = stems if sample_names is None else sample_names & stems
    return sorted(sample_names or [])


def extract_sample_name(filename: str):
    if filename.startswith("sample_"):
        parts = filename.split("_")
        if len(parts) >= 2:
            return f"{parts[0]}_{parts[1]}"
    return Path(filename).stem


def find_sample_path(folder_dir: Path, sample_name: str):
    matches = sorted(folder_dir.glob(f"{sample_name}_*.png"))
    if not matches:
        raise FileNotFoundError(f"Missing {sample_name} in {folder_dir}")
    return matches[0]


def load_grayscale(path: Path):
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

    return canvas, {
        "x_off": x_off,
        "y_off": y_off,
        "width": new_w,
        "height": new_h,
    }


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


def extract_zoom(img: Image.Image, meta):
    box = zoom_box_from_meta(meta)
    patch = img.crop(box)
    patch = patch.resize(TARGET_SIZE, Image.Resampling.NEAREST)
    return patch, box


def render_scale_figure(scale: int):
    scale_dir = QUAL_ROOT / f"x{scale}"
    sample_names = list_sample_names(scale_dir)
    if not sample_names:
        raise RuntimeError(f"No shared samples found in {scale_dir}")

    rows = len(sample_names) * 2
    cols = len(MODEL_ORDER)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.4, rows * 2.0), dpi=300)
    axes = np.atleast_2d(axes)
    plt.subplots_adjust(wspace=0.02, hspace=0.04)

    for sample_idx, sample_name in enumerate(sample_names):
        top_row = sample_idx * 2
        zoom_row = top_row + 1

        for col_idx, (folder, title) in enumerate(MODEL_ORDER):
            img_path = find_sample_path(scale_dir / folder, sample_name)
            raw = load_grayscale(img_path)
            aligned, meta = fit_to_canvas(raw, TARGET_SIZE)
            patch, box = extract_zoom(aligned, meta)

            axes[top_row, col_idx].imshow(aligned, cmap="gray", vmin=0, vmax=255)
            axes[top_row, col_idx].axis("off")
            if sample_idx == 0:
                axes[top_row, col_idx].set_title(title, fontsize=11, fontweight="bold")

            rect = plt.Rectangle(
                (box[0], box[1]),
                box[2] - box[0],
                box[3] - box[1],
                edgecolor="cyan",
                facecolor="none",
                linewidth=1.5,
            )
            axes[top_row, col_idx].add_patch(rect)

            axes[zoom_row, col_idx].imshow(patch, cmap="gray", vmin=0, vmax=255)
            axes[zoom_row, col_idx].axis("off")
            rect_zoom = plt.Rectangle(
                (0, 0),
                TARGET_SIZE[0] - 1,
                TARGET_SIZE[1] - 1,
                edgecolor="cyan",
                facecolor="none",
                linewidth=2.0,
            )
            axes[zoom_row, col_idx].add_patch(rect_zoom)

        axes[top_row, 0].text(
            -0.04,
            0.5,
            sample_name.replace("_", " "),
            transform=axes[top_row, 0].transAxes,
            rotation=90,
            va="center",
            ha="right",
            fontsize=10,
            fontweight="bold",
        )

    OUTPUT_ROOT.mkdir(exist_ok=True)
    out_path = OUTPUT_ROOT / f"Aligned_Qualitative_x{scale}.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved {out_path}")


if __name__ == "__main__":
    render_scale_figure(4)
    render_scale_figure(2)
