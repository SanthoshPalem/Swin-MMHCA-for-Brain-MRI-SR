import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import time
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import lpips
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
import matplotlib.pyplot as plt
import numpy as np

from src.models.swin_mmhca import SwinMMHCA
from src.data.dataloader import MultiModalSuperResDataset

# --- 1. PatchGAN Discriminator ---
class Discriminator(nn.Module):
    def __init__(self, in_channels=1):
        super(Discriminator, self).__init__()
        def block(in_f, out_f, stride=2):
            return nn.Sequential(
                nn.Conv2d(in_f, out_f, 4, stride, 1, bias=False),
                nn.BatchNorm2d(out_f),
                nn.LeakyReLU(0.2, inplace=True)
            )
        self.model = nn.Sequential(
            block(in_channels, 64),
            block(64, 128),
            block(128, 256),
            block(256, 512, stride=1),
            nn.Conv2d(512, 1, 4, 1, 1)
        )
    def forward(self, x): return self.model(x)

# --- 2. Optimized Edge Loss ---
class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        self.register_buffer('fh', torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).float().view(1, 1, 3, 3))
        self.register_buffer('fv', torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).float().view(1, 1, 3, 3))

    def forward(self, pred, target):
        grad_pred = torch.sqrt(F.conv2d(pred, self.fh, padding=1)**2 + F.conv2d(pred, self.fv, padding=1)**2 + 1e-6)
        grad_target = torch.sqrt(F.conv2d(target, self.fh, padding=1)**2 + F.conv2d(target, self.fv, padding=1)**2 + 1e-6)
        return F.l1_loss(grad_pred, grad_target)

def save_visual_comparison(epoch, model, dataloader, device, save_dir, args):
    """Saves visual comparisons with robust normalization."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model.eval()
    with torch.no_grad():
        for i, (lr_images, hr_image) in enumerate(dataloader):
            if i >= 3: break
            lr_imgs = [img.to(device) for img in lr_images]
            hr_img = hr_image.to(device)

            outputs_sr, _, _ = model(lr_imgs)
            
            # Diagnostic prints
            print(f"Sample {i+1} SR range: min={outputs_sr.min().item():.4f}, max={outputs_sr.max().item():.4f}")
            print(f"Sample {i+1} HR range: min={hr_img.min().item():.4f}, max={hr_img.max().item():.4f}")

            # Use T2 for LR display (index 1 if n_inputs > 1 else 0)
            idx = 1 if len(lr_imgs) > 1 else 0
            lr_display = lr_imgs[idx].squeeze().cpu().numpy()
            hr_display = hr_img.squeeze().cpu().numpy()
            output_display = outputs_sr.squeeze().cpu().numpy()
            
            # Robust normalization for visualization
            def norm(x):
                x = np.nan_to_num(x)
                if x.max() > x.min():
                    return (x - x.min()) / (x.max() - x.min())
                return x

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle(f'Epoch {epoch} - Sample {i+1}', fontsize=16)
            axes[0].imshow(norm(lr_display), cmap='gray'); axes[0].set_title('LR Input (T2)')
            axes[1].imshow(norm(output_display), cmap='gray'); axes[1].set_title('SR Output')
            axes[2].imshow(norm(hr_display), cmap='gray'); axes[2].set_title('HR Target')
            for ax in axes: ax.axis('off')
            plt.savefig(os.path.join(save_dir, f'sample_{i+1}.png')); plt.close(fig)

    print(f"--- Visualizations saved to {save_dir} ---")
    model.train()

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SwinMMHCA(n_inputs=args.n_inputs, scale=args.scale_factor).to(device)
    netD = Discriminator().to(device)
    
    optimizerG = optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(0.5, 0.999))
    scalerG = GradScaler('cuda'); scalerD = GradScaler('cuda')
    
    criterion_gan = nn.MSELoss()
    criterion_l1 = nn.L1Loss()
    criterion_edge = EdgeLoss().to(device)
    lpips_fn = lpips.LPIPS(net='alex').to(device)

    train_dataset = MultiModalSuperResDataset(args.dataset_root, split='train', transform=ToTensor(), scale_factor=args.scale_factor)
    dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, persistent_workers=True)
    
    val_dataset = MultiModalSuperResDataset(args.dataset_root, split='test', transform=ToTensor(), scale_factor=args.scale_factor, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    for epoch in range(args.epochs):
        model.train()
        # Adversarial Warming: Introduce GAN loss gradually after epoch 10
        adv_weight = 0.1 if epoch >= 10 else 0.0
        
        for i, (lr_imgs, hr_img) in enumerate(dataloader):
            lr_imgs = [img.to(device) for img in lr_imgs]; hr_img = hr_img.to(device)
            
            # --- Train Discriminator ---
            optimizerD.zero_grad()
            with autocast('cuda'):
                real_out = netD(hr_img)
                loss_real = criterion_gan(real_out, torch.ones_like(real_out))
                fake_hr, _, _ = model(lr_imgs)
                fake_out = netD(fake_hr.detach())
                loss_fake = criterion_gan(fake_out, torch.zeros_like(fake_out))
                loss_D = (loss_real + loss_fake) * 0.5
            scalerD.scale(loss_D).backward(); scalerD.step(optimizerD); scalerD.update()

            # --- Train Generator ---
            optimizerG.zero_grad()
            with autocast('cuda'):
                fake_hr, seg_mask, _ = model(lr_imgs)
                loss_adv = criterion_gan(netD(fake_hr), torch.ones_like(real_out))
                loss_l1 = criterion_l1(fake_hr, hr_img)
                loss_perc = lpips_fn(fake_hr * 2 - 1, hr_img * 2 - 1).mean()
                loss_edge = criterion_edge(fake_hr, hr_img)
                loss_seg = F.binary_cross_entropy_with_logits(seg_mask, (hr_img > 0.5).float())
                
                loss_G = 10.0 * loss_l1 + 1.0 * loss_perc + 2.0 * loss_edge + adv_weight * loss_adv + 0.1 * loss_seg
            
            scalerG.scale(loss_G).backward(); scalerG.step(optimizerG); scalerG.update()
            if (i+1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}] Step [{i+1}/{len(dataloader)}] LossG: {loss_G.item():.4f} LossD: {loss_D.item():.4f} AdvW: {adv_weight}")

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"epoch_checkpoints/gan_epoch_{epoch+1}.pth")
            save_visual_comparison(epoch + 1, model, val_dataloader, device, f"epoch_visuals/epoch_{epoch+1}_visuals", args)

    if args.save_checkpoint:
        torch.save(model.state_dict(), args.save_checkpoint)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--dataset_root', type=str, default='datasets')
    parser.add_argument('--scale_factor', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--n_inputs', type=int, default=3)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--save_checkpoint', type=str, default=None)
    train(parser.parse_args())
