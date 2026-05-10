import os
import torch
from pathlib import Path

def filter_golden_slices(
    dataset_dir="processed_data",
    min_slice=35,
    max_slice=95,
    pixel_threshold=0.10,
    coverage_threshold=0.28,
):
    """
    Two-stage dataset filter:

    Stage 1 — Index Gate (fast, no file I/O):
        Deletes slices whose index falls outside the 'Golden Zone' [min_slice, max_slice].
        These are the clear top/bottom skull slices.

    Stage 2 — Brain Coverage Filter (catches what Stage 1 misses):
        Some slices inside the index range still show inferior brain anatomy
        (cerebellum, brainstem) with a small brain cross-section in a large
        black background. We reject these by checking:

            coverage = (pixels > pixel_threshold) / total_pixels

        If coverage < coverage_threshold the slice is deleted.
        Good cortical slices: ~35–65% coverage.
        Bad inferior slices:  ~10–25% coverage.
    """
    dataset_path = Path(dataset_dir)

    if not dataset_path.exists():
        print(f"Error: dataset path '{dataset_path}' does not exist.")
        return

    pt_files = list(dataset_path.glob("*.pt"))
    total_files = len(pt_files)
    print(f"Found {total_files} slices to scan.\n")

    deleted_out_of_range = 0
    deleted_low_coverage = 0

    for pt_file in pt_files:
        filename = pt_file.name

        # ── Stage 1: Index Gate ──────────────────────────────────────────────
        try:
            slice_str = filename.split("_slice_")[-1].replace(".pt", "")
            slice_idx = int(slice_str)
        except Exception:
            print(f"  [skip] Cannot parse slice index from '{filename}'.")
            continue

        if slice_idx < min_slice or slice_idx > max_slice:
            os.remove(str(pt_file))
            deleted_out_of_range += 1
            continue

        # ── Stage 2: Brain Coverage Filter ──────────────────────────────────
        try:
            tensor = torch.load(pt_file, weights_only=True)  # [3, H, W]
            t2 = tensor[1]  # T2 channel — best grey/white matter contrast

            total_pixels = t2.numel()
            brain_pixels = (t2 > pixel_threshold).sum().item()
            coverage = brain_pixels / total_pixels

            if coverage < coverage_threshold:
                os.remove(str(pt_file))
                deleted_low_coverage += 1

        except Exception as e:
            print(f"  [error] Could not read '{filename}': {e}")

    total_deleted = deleted_out_of_range + deleted_low_coverage
    remaining = total_files - total_deleted

    print("=" * 60)
    print("  🧠 DATASET SURGERY: COMPLETE")
    print("=" * 60)
    print(f"  Deleted (outside index range {min_slice}–{max_slice}): {deleted_out_of_range}")
    print(f"  Deleted (inferior slice / low brain coverage):  {deleted_low_coverage}")
    print(f"  Total deleted:                                  {total_deleted}")
    print(f"  Remaining high-quality cortical slices:         {remaining}")
    print("=" * 60)


if __name__ == "__main__":
    print("=" * 60)
    print("  🧠 DATASET SURGERY: ISOLATING THE GOLDEN ZONE 🧠")
    print("=" * 60)

    # Stage 1: keep only slices 35–95 (core brain anatomy)
    # Stage 2: additionally remove inferior slices with < 28% brain coverage
    filter_golden_slices(
        min_slice=35,
        max_slice=95,
        pixel_threshold=0.10,
        coverage_threshold=0.28,
    )
    print("\n✅ Done. The model will now train exclusively on high-quality cortical slices.")
