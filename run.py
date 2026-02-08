import argparse
import os
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import time
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import lpips
import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

from src.models.swin_mmhca import SwinMMHCA
from src.data.dataloader import MultiModalSuperResDataset
from src.models.options import get_args
from src.models.edsr_nav import EDSR_Nav

def save_visual_comparison(epoch, model, dataloader, device, save_dir, args):
    """Saves visual comparisons of model output for a few validation samples."""
    print(f"--- Generating visualizations for epoch {epoch} ---")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model.eval()
    with torch.no_grad():
        for i, (lr_images, hr_image) in enumerate(dataloader):
            if i >= args.num_samples:
                break

            if args.n_inputs == 1:
                lr_images = lr_images[0].unsqueeze(0)

            lr_images = [img.to(device) for img in lr_images]
            hr_image = hr_image.to(device)

            # Generate model output
            outputs = model(lr_images if args.n_inputs > 1 else lr_images[0])

            # Prepare images for display
            # Use T2 modality for LR input display
            lr_display = lr_images[1].squeeze().cpu().numpy()
            hr_display = hr_image.squeeze().cpu().numpy()
            output_display = outputs.squeeze().cpu().numpy()

            # Plot and save the comparison
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle(f'Epoch {epoch} - Sample {i+1}', fontsize=16)

            axes[0].imshow(lr_display, cmap='gray')
            axes[0].set_title('Low-Resolution Input (T2)')
            axes[0].axis('off')

            axes[1].imshow(output_display, cmap='gray')
            axes[1].set_title('Super-Resolved Output')
            axes[1].axis('off')

            axes[2].imshow(hr_display, cmap='gray')
            axes[2].set_title('High-Resolution Ground Truth')
            axes[2].axis('off')

            plt.tight_layout()
            save_path = os.path.join(save_dir, f'comparison_sample_{i+1}.png')
            plt.savefig(save_path)
            plt.close(fig)

    print(f"--- Visualizations saved to {save_dir} ---")
    model.train()


def train(args):
    start_time = time.time()

    # --- Setup device and model parallelization ---
    device = torch.device('cuda' if not args.cpu and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = SwinMMHCA(n_inputs=args.n_inputs, scale=args.scale_factor)
    
    # Enable DataParallel for multi-GPU usage
    if torch.cuda.device_count() > 1 and args.n_GPUs > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    
    model.to(device)

    # --- Setup datasets and dataloaders ---
    if args.n_inputs == 1:
        modalities = ['T2']
    else:
        modalities = ['T1', 'T2', 'PD']

    train_dataset = MultiModalSuperResDataset(
        dataset_root=args.dataset_root,
        modalities=modalities,
        scale_factor=args.scale_factor,
        transform=ToTensor(),
        split='train'
    )
    dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    # Create a separate dataloader for validation/visualization
    val_dataset = MultiModalSuperResDataset(
        dataset_root=args.dataset_root,
        modalities=modalities,
        scale_factor=args.scale_factor,
        transform=ToTensor(),
        split='test',
        shuffle=False # No need to shuffle for visualization
    )
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    # --- Setup optimizer, loss, and mixed-precision scaler ---
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scaler = GradScaler(enabled=not args.cpu and torch.cuda.is_available())

    start_epoch = 0
    # --- Load from checkpoint if resuming ---
    if args.resume_from and os.path.exists(args.resume_from):
        print(f"Loading checkpoint: {args.resume_from}")
        checkpoint = torch.load(args.resume_from, map_location=device)
        
        state_dict_to_load = None
        # Check if it's a full checkpoint or just a state_dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            print("Resuming training from full checkpoint.")
            state_dict_to_load = checkpoint['model_state_dict']
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1
            print(f"Resuming from epoch {start_epoch}")
        else:
            # This is a raw state_dict from an epoch-wise save
            print("Loaded model weights only. Optimizer will be reset.")
            state_dict_to_load = checkpoint
            # Try to infer epoch from filename
            match = re.search(r'_epoch_(\d+)\.pth$', args.resume_from)
            if match:
                start_epoch = int(match.group(1))
                print(f"Inferred start epoch from filename: {start_epoch}. Resuming training from next epoch.")
            else:
                 print("Could not infer epoch from filename. Starting from epoch 0.")

        # Now, load the state_dict, handling DataParallel prefix
        if state_dict_to_load:
            is_model_dataparallel = isinstance(model, nn.DataParallel)
            is_checkpoint_dataparallel = any(k.startswith('module.') for k in state_dict_to_load.keys())

            if is_model_dataparallel and not is_checkpoint_dataparallel:
                # Current model is DP, checkpoint is not. Add 'module.' prefix.
                print("Adding 'module.' prefix to checkpoint keys for DataParallel model.")
                new_state_dict = {'module.' + k: v for k, v in state_dict_to_load.items()}
                model.load_state_dict(new_state_dict)
            elif not is_model_dataparallel and is_checkpoint_dataparallel:
                # Current model is not DP, checkpoint is. Remove 'module.' prefix.
                print("Removing 'module.' prefix from checkpoint keys for single-GPU model.")
                new_state_dict = {k.replace('module.', ''): v for k, v in state_dict_to_load.items()}
                model.load_state_dict(new_state_dict)
            else:
                # Both are DP or both are not. Load as is.
                model.load_state_dict(state_dict_to_load)

    # --- Create directories for checkpoints and visuals ---
    epoch_save_dir = "epoch_checkpoints"
    os.makedirs(epoch_save_dir, exist_ok=True)
    visual_save_root = "epoch_visuals"
    os.makedirs(visual_save_root, exist_ok=True)

    # --- Training Loop ---
    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()
        model.train()
        running_loss = 0.0
        
        for i, (lr_images, hr_image) in enumerate(dataloader):
            if args.n_inputs == 1:
                lr_images = lr_images[0].unsqueeze(0)
            
            lr_images = [img.to(device, non_blocking=True) for img in lr_images]
            hr_image = hr_image.to(device, non_blocking=True)

            optimizer.zero_grad()
            
            # Use mixed precision
            with autocast(enabled=not args.cpu):
                outputs = model(lr_images if args.n_inputs > 1 else lr_images[0])
                loss = criterion(outputs, hr_image)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

            if (i + 1) % args.log_interval == 0:
                print(f'Epoch [{epoch+1}/{args.epochs}], Step [{i+1}/{len(dataloader)}], Loss: {running_loss / args.log_interval:.4f}')
                running_loss = 0.0

        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch {epoch+1} completed in {epoch_duration:.2f} seconds.")

        # --- Checkpoint and Visualize every 10 epochs ---
        if (epoch + 1) % 10 == 0:
            # 1. Save the model checkpoint
            model_to_save = model.module if isinstance(model, nn.DataParallel) else model
            model_path = os.path.join(epoch_save_dir, f'swin_mmhca_x{args.scale_factor}_epoch_{epoch+1}.pth')
            torch.save(model_to_save.state_dict(), model_path)
            print(f"\nSaved checkpoint to {model_path}")

            # 2. Perform Qualitative Visualization without interrupting
            visual_save_dir = os.path.join(visual_save_root, f'epoch_{epoch+1}_visuals')
            save_visual_comparison(epoch + 1, model, val_dataloader, device, visual_save_dir, args)

    # --- Save final resumable checkpoint ---
    if args.save_checkpoint:
        os.makedirs(os.path.dirname(args.save_checkpoint), exist_ok=True)
        model_to_save = model.module if isinstance(model, nn.DataParallel) else model
        torch.save({
            'epoch': args.epochs - 1,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, args.save_checkpoint)
        print(f"Saved resumable checkpoint to {args.save_checkpoint}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total training time for {args.epochs - start_epoch} epochs: {elapsed_time:.2f} seconds")


def evaluate(args):
    device = torch.device('cuda' if not args.cpu and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if args.model_type == 'EDSR_Nav':
        modalities = ['T2', 'PD']
        n_inputs = 2
    else:
        n_inputs = 3 if args.n_inputs > 1 else 1
        modalities = ['T1', 'T2', 'PD'] if n_inputs > 1 else ['T2']
        
    val_dataset = MultiModalSuperResDataset(
        dataset_root=args.dataset_root,
        modalities=modalities,
        scale_factor=args.scale_factor,
        transform=ToTensor(),
        split='test',
        shuffle=False
    )
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    if args.model_type == 'SwinMMHCA':
        model = SwinMMHCA(n_inputs=n_inputs, scale=args.scale_factor)
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    elif args.model_type == 'EDSR_Nav':
        edsr_args = get_args()
        edsr_args.scale = [args.scale_factor]
        model = EDSR_Nav(edsr_args)
        model.load_state_dict(torch.load(args.edsr_checkpoint_path, map_location=device))
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    model.to(device)
    model.eval()

    psnr = PeakSignalNoiseRatio().to(device)
    ssim = StructuralSimilarityIndexMeasure().to(device)
    lpips_fn = lpips.LPIPS(net='alex').to(device)

    total_psnr, total_ssim, total_lpips = 0.0, 0.0, 0.0

    with torch.no_grad():
        for lr_images, hr_image in val_dataloader:
            if n_inputs == 1:
                lr_images = lr_images[0].unsqueeze(0)
            
            lr_images = [img.to(device) for img in lr_images]
            hr_image = hr_image.to(device)

            outputs = model(lr_images if n_inputs > 1 else lr_images[0])

            total_psnr += psnr(outputs, hr_image)
            total_ssim += ssim(outputs, hr_image)
            # LPIPS expects input range [-1, 1]
            total_lpips += lpips_fn(outputs * 2 - 1, hr_image * 2 - 1).mean()

    avg_psnr = total_psnr / len(val_dataloader)
    avg_ssim = total_ssim / len(val_dataloader)
    avg_lpips = total_lpips / len(val_dataloader)

    print(f'Results for {args.model_type}:')
    print(f'  Average PSNR: {avg_psnr:.4f}')
    print(f'  Average SSIM: {avg_ssim:.4f}')
    print(f'  Average LPIPS: {avg_lpips:.4f}')

def main():
    parser = argparse.ArgumentParser(description='SwinMMHCA for Super-Resolution')

    # Mode
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate'], help='Operating mode')

    # Hardware specifications
    parser.add_argument('--cpu', action='store_true', help='use cpu only')
    parser.add_argument('--n_GPUs', type=int, default=1, help='number of GPUs to use')

    # Data specifications
    parser.add_argument('--dataset_root', type=str, default='datasets', help='root directory of the dataset')
    parser.add_argument('--scale_factor', type=int, default=4, help='super-resolution scale factor')
    parser.add_argument('--n_inputs', type=int, default=3, help='number of input modalities (1 for single-input, >1 for multi-input)')

    # Model specifications
    parser.add_argument('--model_type', type=str, default='SwinMMHCA', choices=['SwinMMHCA', 'EDSR_Nav'], help='type of model to use')
    parser.add_argument('--checkpoint_path', type=str, default='pretrained_models/swin_mmhca.pth', help='path to the model checkpoint')

    # Training specifications
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers for data loading (set to 0 for debugging, >0 for performance)')
    parser.add_argument('--log_interval', type=int, default=10, help='interval for printing training loss')
    parser.add_argument('--save_dir', type=str, default='results', help='directory to save the trained model')
    parser.add_argument('--num_samples', type=int, default=3, help='number of samples to visualize')

    # Resumable training specifications
    parser.add_argument('--save_checkpoint', type=str, default=None, help='Path to save the training state for resuming.')
    parser.add_argument('--resume_from', type=str, default=None, help='Path to a checkpoint to resume training from.')
    
    # Evaluation specifications (only for evaluate mode)
    parser.add_argument('--edsr_checkpoint_path', type=str, default='../MHCA-main/edsr/pretrained_models/model_multi_input_IXI_x4.pt', help='Path to the EDSR_Nav model checkpoint for comparison')

    args = parser.parse_args()

    # Automatically adjust n_GPUs if more are available
    if not args.cpu and torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        if gpu_count > 1:
            print(f"Found {gpu_count} GPUs available.")
            if args.n_GPUs < gpu_count:
                print(f"Increasing n_GPUs to {gpu_count} to utilize all available resources.")
                args.n_GPUs = gpu_count
    
    if args.mode == 'train':
        train(args)
    elif args.mode == 'evaluate':
        evaluate(args)

if __name__ == '__main__':
    main()