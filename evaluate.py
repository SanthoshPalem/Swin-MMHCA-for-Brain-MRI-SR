import argparse
import csv
import os
import random
import sys

import lpips
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from src.data.dataloader import PreprocessedSuperResDataset
from src.models.swin_mmhca import SwinMMHCA
from src.utils.metrics import calculate_psnr, calculate_ssim, crop_border


def save_visual(output_dir, sample_idx, lr_image, sr_image, hr_image):
    os.makedirs(output_dir, exist_ok=True)

    lr_display = lr_image.squeeze().cpu().numpy()
    sr_display = sr_image.squeeze().cpu().numpy()
    hr_display = hr_image.squeeze().cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(lr_display, cmap="gray")
    axes[0].set_title("LR Input (T2)")
    axes[1].imshow(sr_display, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("SR Output")
    axes[2].imshow(hr_display, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("HR Target")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"sample_{sample_idx:04d}.png"))
    plt.close(fig)


def save_metrics(output_path, split, stage, psnr, ssim, lpips_value):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Split", "Stage", "PSNR", "SSIM", "LPIPS"])
        writer.writerow([split, stage, f"{psnr:.6f}", f"{ssim:.6f}", f"{lpips_value:.6f}"])


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Evaluating split: {args.split}")
    print(f"Checkpoint: {args.checkpoint_path}")

    dataset = PreprocessedSuperResDataset(
        processed_dir=args.dataset_root,
        split=args.split,
        scale_factor=args.scale_factor,
        shuffle=False,
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    print(f"Samples: {len(dataset)}")

    visual_indices = set()
    if args.save_visuals_dir:
        max_visuals = min(args.max_visuals, len(dataset))
        if args.random_visuals:
            rng = random.Random(args.visual_seed)
            visual_indices = set(rng.sample(range(len(dataset)), max_visuals))
            print(f"Saving visuals for {len(visual_indices)} random {args.split} samples with seed {args.visual_seed}")
        else:
            visual_indices = set(range(max_visuals))

    h_w = 128 if args.scale_factor == 2 else 64
    model = SwinMMHCA(n_inputs=args.n_inputs, scale=args.scale_factor, height=h_w, width=h_w).to(device)

    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        stage = checkpoint.get("stage", args.training_stage)
    else:
        model.load_state_dict(checkpoint, strict=False)
        stage = args.training_stage

    model.eval()

    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_fn = lpips.LPIPS(net="alex").to(device)

    total_psnr = 0.0
    total_ssim = 0.0
    total_lpips = 0.0

    with torch.no_grad():
        saved_visuals = 0
        for i, (lr_images, hr_image) in enumerate(dataloader, start=1):
            lr_imgs = [img.to(device) for img in lr_images]
            hr_image = hr_image.to(device)

            outputs = model(lr_imgs, stage=stage)
            sr = outputs["sr"]

            total_psnr += calculate_psnr(sr, hr_image, scale=args.scale_factor, metric=psnr_metric).item()
            total_ssim += calculate_ssim(sr, hr_image, scale=args.scale_factor, metric=ssim_metric).item()
            cropped_sr, cropped_hr = crop_border(sr, hr_image, scale=args.scale_factor)
            total_lpips += lpips_fn(cropped_sr * 2.0 - 1.0, cropped_hr * 2.0 - 1.0).mean().item()

            dataset_index = i - 1
            if args.save_visuals_dir and dataset_index in visual_indices:
                lr_for_display = lr_imgs[1 if len(lr_imgs) > 1 else 0]
                save_visual(args.save_visuals_dir, dataset_index + 1, lr_for_display, sr, hr_image)
                saved_visuals += 1

            if i % 200 == 0:
                print(f"Processed {i}/{len(dataloader)} samples...")

    denom = max(1, len(dataloader))
    avg_psnr = total_psnr / denom
    avg_ssim = total_ssim / denom
    avg_lpips = total_lpips / denom
    metrics_path = args.metrics_out or os.path.join("results", f"{args.split}_metrics_stage_{stage}.csv")
    save_metrics(metrics_path, args.split, stage, avg_psnr, avg_ssim, avg_lpips)

    print(f"\n{'=' * 50}")
    print(f"Evaluation Results [{args.split.upper()} SET]")
    print(f"{'=' * 50}")
    print(f"Stage : {stage}")
    print(f"PSNR  : {avg_psnr:.4f}")
    print(f"SSIM  : {avg_ssim:.4f}")
    print(f"LPIPS : {avg_lpips:.4f}")
    print("PSNR is computed on [0,1]-normalized images with border cropping matched to ground truth.")
    print(f"Metrics saved to: {metrics_path}")
    if args.save_visuals_dir:
        print(f"Saved {saved_visuals} visual comparisons to: {args.save_visuals_dir}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SwinMMHCA on preprocessed dataset")
    parser.add_argument("--dataset_root", type=str, default="processed_data")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--scale_factor", type=int, default=4)
    parser.add_argument("--n_inputs", type=int, default=3)
    parser.add_argument("--training_stage", type=int, default=4, choices=[1, 2, 3, 4])
    parser.add_argument("--split", type=str, default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--save_visuals_dir", type=str, default=None)
    parser.add_argument("--max_visuals", type=int, default=100)
    parser.add_argument("--metrics_out", type=str, default=None)
    parser.add_argument("--random_visuals", action="store_true")
    parser.add_argument("--visual_seed", type=int, default=42)
    evaluate(parser.parse_args())
