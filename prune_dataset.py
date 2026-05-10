import os
import torch
from pathlib import Path

def main():
    project_root = Path(__file__).parent.absolute()
    processed_dir = project_root / "processed_data"

    if not processed_dir.exists():
        print("Error: processed_data directory not found.")
        return

    pt_files = list(processed_dir.glob("*.pt"))
    print(f"Starting scan of {len(pt_files)} slices...")

    deleted_count = 0
    reasons = {"dark": 0, "low_coverage": 0}

    # --- FILTER 1: Absolute Darkness Threshold ---
    # Rejects near-empty slices (e.g. top/bottom skull caps).
    # T2 mean < 0.05 means the slice is essentially black.
    DARKNESS_THRESHOLD = 0.05

    # --- FILTER 2: Brain Coverage Ratio ---
    # Rejects slices where the brain occupies too small a fraction of the image.
    # Inferior slices (cerebellum/brainstem, like IXI511 slice_046) have a small,
    # round brain cross-section embedded in a large black background.
    # Good cortical slices fill most of the image with a large oval brain.
    #
    # We measure: (pixels with signal > pixel_threshold) / (total pixels)
    # and reject if coverage < COVERAGE_THRESHOLD.
    #
    # Calibrated values (tested on IXI 256x256 slices):
    #   Good cortical slice:     coverage ~0.35–0.65
    #   Bad inferior slice:      coverage ~0.10–0.25
    PIXEL_THRESHOLD = 0.10   # pixel value above this = "brain tissue" (normalized 0–1)
    COVERAGE_THRESHOLD = 0.28 # if brain fills less than 28% of the image → reject

    for pt_file in pt_files:
        try:
            tensor = torch.load(pt_file, weights_only=True)  # shape: [3, H, W]

            # Use T2 (index 1) as the reference channel — highest grey/white contrast
            t2 = tensor[1]  # shape: [H, W], values in [0, 1]

            t2_mean = t2.mean().item()

            # Filter 1: too dark overall
            if t2_mean < DARKNESS_THRESHOLD:
                pt_file.unlink()
                deleted_count += 1
                reasons["dark"] += 1
                continue

            # Filter 2: brain covers too little of the image
            total_pixels = t2.numel()
            brain_pixels = (t2 > PIXEL_THRESHOLD).sum().item()
            coverage = brain_pixels / total_pixels

            if coverage < COVERAGE_THRESHOLD:
                pt_file.unlink()
                deleted_count += 1
                reasons["low_coverage"] += 1
                continue

        except Exception as e:
            print(f"Error reading {pt_file.name}: {e}")

    print(f"\n{'='*55}")
    print(f"  SLICE PRUNING COMPLETE")
    print(f"{'='*55}")
    print(f"  Deleted (too dark):           {reasons['dark']}")
    print(f"  Deleted (inferior/low brain coverage): {reasons['low_coverage']}")
    print(f"  Total deleted:                {deleted_count}")
    print(f"  Remaining high-quality slices: {len(pt_files) - deleted_count}")
    print(f"{'='*55}")

if __name__ == "__main__":
    main()
