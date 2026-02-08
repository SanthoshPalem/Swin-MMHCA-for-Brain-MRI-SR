
import argparse
import torch
import os
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import lpips
import numpy as np

from src.models.swin_mmhca import SwinMMHCA
from src.data.dataloader import MultiModalSuperResDataset

def evaluate_model(model, dataloader, device, n_inputs):
    """Evaluates a single model and returns the average metrics."""
    model.eval()
    psnr_metric = PeakSignalNoiseRatio().to(device)
    ssim_metric = StructuralSimilarityIndexMeasure().to(device)
    lpips_metric = lpips.LPIPS(net='alex').to(device)

    total_psnr, total_ssim, total_lpips = 0.0, 0.0, 0.0
    with torch.no_grad():
        for lr_images, hr_image in dataloader:
            if n_inputs == 1:
                lr_images = [lr_images[0].unsqueeze(0)]
            
            lr_images = [img.to(device) for img in lr_images]
            hr_image = hr_image.to(device)

            outputs = model(lr_images if n_inputs > 1 else lr_images[0])

            # Clamp outputs to [0, 1] range for metrics
            outputs_clamped = torch.clamp(outputs, 0, 1)

            total_psnr += psnr_metric(outputs_clamped, hr_image)
            total_ssim += ssim_metric(outputs_clamped, hr_image)
            # LPIPS expects input range [-1, 1]
            total_lpips += lpips_metric(outputs_clamped * 2 - 1, hr_image * 2 - 1).mean()

    num_batches = len(dataloader)
    return (total_psnr / num_batches, total_ssim / num_batches, total_lpips / num_batches)

def main(args):
    device = torch.device('cuda' if not args.cpu and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Load Dataset ---
    modalities = ['T1', 'T2', 'PD'] if args.n_inputs > 1 else ['T2']
    val_dataset = MultiModalSuperResDataset(
        dataset_root=args.dataset_root,
        modalities=modalities,
        scale_factor=args.scale_factor,
        transform=ToTensor(),
        split='test',
        shuffle=False
    )
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # --- Load Model 1 ---
    print(f"Loading Model 1 from: {args.checkpoint1_path}")
    model1 = SwinMMHCA(n_inputs=args.n_inputs, scale=args.scale_factor).to(device)
    model1.load_state_dict(torch.load(args.checkpoint1_path, map_location=device))

    # --- Load Model 2 ---
    print(f"Loading Model 2 from: {args.checkpoint2_path}")
    model2 = SwinMMHCA(n_inputs=args.n_inputs, scale=args.scale_factor).to(device)
    model2.load_state_dict(torch.load(args.checkpoint2_path, map_location=device))
    
    # --- Evaluate Both Models ---
    print("\nEvaluating Model 1...")
    psnr1, ssim1, lpips1 = evaluate_model(model1, val_dataloader, device, args.n_inputs)
    
    print("\nEvaluating Model 2...")
    psnr2, ssim2, lpips2 = evaluate_model(model2, val_dataloader, device, args.n_inputs)

    # --- Print Results ---
    print("\n" + "="*50)
    print("CONVERGENCE CHECK RESULTS")
    print("="*50)
    print(f"Model 1 ({os.path.basename(args.checkpoint1_path)}):")
    print(f"  PSNR:  {psnr1:.4f} (Higher is better)")
    print(f"  SSIM:  {ssim1:.4f} (Higher is better)")
    print(f"  LPIPS: {lpips1:.4f} (Lower is better)")
    print("\n")
    print(f"Model 2 ({os.path.basename(args.checkpoint2_path)}):")
    print(f"  PSNR:  {psnr2:.4f}")
    print(f"  SSIM:  {ssim2:.4f}")
    print(f"  LPIPS: {lpips2:.4f}")
    print("="*50)

    # --- Conclusion ---
    psnr_diff = psnr2 - psnr1
    ssim_diff = ssim2 - ssim1
    lpips_diff = lpips1 - lpips2 # Lower is better, so diff is p1 - p2

    print("\nAnalysis:")
    print(f"Change in PSNR:  {psnr_diff:+.6f}")
    print(f"Change in SSIM:  {ssim_diff:+.6f}")
    print(f"Change in LPIPS: {lpips_diff:+.6f}")

    if abs(psnr_diff) < 0.01 and abs(ssim_diff) < 0.001 and abs(lpips_diff) < 0.001:
        print("\nConclusion: The model has likely CONVERGED.")
        print("The changes in performance are minimal. Further training may not be beneficial.")
    else:
        print("\nConclusion: The model is likely STILL LEARNING.")
        print("There are still noticeable changes in performance between these epochs.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convergence Check for SwinMMHCA')
    parser.add_argument('--checkpoint1_path', type=str, required=True, help='Path to the first (earlier) model checkpoint.')
    parser.add_argument('--checkpoint2_path', type=str, required=True, help='Path to the second (later) model checkpoint.')
    
    # Add other necessary args from run.py with defaults
    parser.add_argument('--dataset_root', type=str, default='datasets', help='Root directory of the dataset')
    parser.add_argument('--scale_factor', type=int, default=4, help='Super-resolution scale factor')
    parser.add_argument('--n_inputs', type=int, default=3, help='Number of input modalities')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for evaluation')
    parser.add_argument('--cpu', action='store_true', help='Use CPU only')
    
    args = parser.parse_args()
    main(args)
