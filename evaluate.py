import argparse
import os
import torch
from torch.utils.data import DataLoader
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import lpips
import sys

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
        transform=None, # ToTensor is handled inside logic if needed, but run.py uses ToTensor in transform.
                        # Wait, run.py uses transform=ToTensor().
                        # Let's check dataloader again. 
                        # Dataloader returns PIL if transform is None, Tensor if ToTensor.
                        # We need Tensor.
        split='test',
        shuffle=False
    )
    
    # We need to import ToTensor
    from torchvision.transforms import ToTensor
    val_dataset.transform = ToTensor()

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
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        raise FileNotFoundError(f"Checkpoint not found at {args.checkpoint_path}")

    model.eval()

    # --- Metrics ---
    psnr_fn = PeakSignalNoiseRatio().to(device)
    ssim_fn = StructuralSimilarityIndexMeasure().to(device)
    lpips_fn = lpips.LPIPS(net='alex').to(device)

    total_psnr = 0.0
    total_ssim = 0.0
    total_lpips = 0.0

    with torch.no_grad():
        for i, (lr_images, hr_image) in enumerate(val_dataloader):
            if args.n_inputs == 1:
                # If list, take first element and unsqueeze if needed (dataloader returns list of images)
                # Dataloader returns lr_images as a list of tensors.
                lr_images = lr_images[0].unsqueeze(0) 
                # Wait, dataloader returns list of 3D tensors (C, H, W) if batch size is involved?
                # Actually dataloader returns: lr_images (list of tensors), hr_image (tensor)
                # If batch_size > 1, lr_images is a list of [B, C, H, W] tensors.
                
                # In run.py: 
                # if args.n_inputs == 1: lr_images = lr_images[0].unsqueeze(0) -> this looks suspicious in run.py if batching is used
                # actually run.py says: lr_images = lr_images[0].unsqueeze(0)
                # Let's double check dataloader output.
                # __getitem__ returns lr_images (list of PIL transformed), hr_image.
                # DataLoader collates them.
                # lr_images will be a list of Tensors [Batch, C, H, W].
                pass

            # Correct handling of input list
            if isinstance(lr_images, list):
                 lr_images = [img.to(device) for img in lr_images]
            else:
                 lr_images = lr_images.to(device)
            
            hr_image = hr_image.to(device)

            # Forward pass
            # SwinMMHCA expects list if multi-input, tensor if single input (based on my read of swin_mmhca.py)
            # swin_mmhca.py: if isinstance(x, list): ... else: ...
            
            if args.n_inputs == 1:
                # If it's a list from dataloader, take the T2 component (index 0 if modalities=['T2'])
                # But if modalities=['T1', 'T2', 'PD'] and we want single input... dataloader logic says:
                # if n_inputs=1, modalities=['T2']. So lr_images is list of length 1.
                outputs = model(lr_images[0])
            else:
                outputs = model(lr_images)

            # Clamp outputs to [0, 1] for metric calculation stability
            outputs = torch.clamp(outputs, 0, 1)
            # hr_image should already be [0, 1] from ToTensor()

            total_psnr += psnr_fn(outputs, hr_image)
            total_ssim += ssim_fn(outputs, hr_image)
            # LPIPS expects inputs in range [-1, 1]
            total_lpips += lpips_fn(outputs * 2 - 1, hr_image * 2 - 1).mean()
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1} batches...")

    avg_psnr = total_psnr / len(val_dataloader)
    avg_ssim = total_ssim / len(val_dataloader)
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
