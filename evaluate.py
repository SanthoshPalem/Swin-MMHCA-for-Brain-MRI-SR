import argparse
import os
import torch
from torch.utils.data import DataLoader
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import lpips
import sys
from torchvision.transforms import ToTensor

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from src.data.dataloader import MultiModalSuperResDataset
from src.models.swin_mmhca import SwinMMHCA

def evaluate(args):
    device = torch.device('cuda' if not args.cpu and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Determine modalities based on inputs
    if args.n_inputs == 1:
        modalities = ['T2']
    else:
        modalities = ['T1', 'T2', 'PD']
    
    print(f"Evaluating with modalities: {modalities}")

    # --- Dataset ---
    val_dataset = MultiModalSuperResDataset(
        dataset_root=args.dataset_root,
        modalities=modalities,
        scale_factor=args.scale_factor,
        transform=ToTensor(),
        split='test',
        shuffle=False
    )

    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # --- Model ---
    model = SwinMMHCA(
        n_inputs=args.n_inputs,
        scale=args.scale_factor
    ).to(device)
    
    if os.path.exists(args.checkpoint_path):
        print(f"Loading checkpoint: {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        
        # Handle if checkpoint is a dict (state_dict + optimizer) or just state_dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
    else:
        raise FileNotFoundError(f"Checkpoint not found at {args.checkpoint_path}")

    model.eval()

    # --- Metrics (Corrected Usage) ---
    psnr_fn = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_fn = lpips.LPIPS(net='alex').to(device)

    total_lpips = 0.0

    with torch.no_grad():
        for i, (lr_images, hr_image) in enumerate(val_dataloader):
            # Correct handling of input list
            if isinstance(lr_images, list):
                 lr_images = [img.to(device) for img in lr_images]
                 input_tensor = lr_images if args.n_inputs > 1 else lr_images[0]
            else:
                 input_tensor = lr_images.to(device)
            
            hr_image = hr_image.to(device)

            # Forward pass
            outputs_sr, _, _ = model(input_tensor)

            # Clamp outputs to [0, 1] for metric calculation stability
            outputs_sr = torch.clamp(outputs_sr, 0, 1)

            # 1. Update torchmetrics (Fixes accumulation issue)
            psnr_fn.update(outputs_sr, hr_image)
            ssim_fn.update(outputs_sr, hr_image)

            # 2. Accumulate LPIPS manually
            # LPIPS expects inputs in range [-1, 1]
            total_lpips += lpips_fn(outputs_sr * 2 - 1, hr_image * 2 - 1).mean().item()
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1} batches...")

    # --- Final Computation ---
    avg_psnr = psnr_fn.compute().item()
    avg_ssim = ssim_fn.compute().item()
    avg_lpips = total_lpips / len(val_dataloader)

    print(f'\nResults for SwinMMHCA (x{args.scale_factor}):')
    print(f'Average PSNR: {avg_psnr:.4f}')
    print(f'Average SSIM: {avg_ssim:.4f}')
    print(f'Average LPIPS: {avg_lpips:.4f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate SwinMMHCA')
    
    parser.add_argument('--dataset_root', type=str, default='datasets', help='Root directory of the dataset')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--scale_factor', type=int, default=4, help='Super-resolution scale factor')
    parser.add_argument('--n_inputs', type=int, default=3, help='Number of input modalities (1 or 3)')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for data loading')
    parser.add_argument('--cpu', action='store_true', help='Use CPU only')

    args = parser.parse_args()
    
    evaluate(args)
