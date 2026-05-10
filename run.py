import argparse
import csv
import os

import lpips
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from src.data.dataloader import PreprocessedSuperResDataset
from src.models.swin_mmhca import SwinMMHCA
from src.utils.metrics import calculate_psnr, calculate_ssim, crop_border


class PatchDiscriminator70(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()

        def block(in_ch, out_ch, stride=2, normalize=True):
            layers = [nn.Conv2d(in_ch, out_ch, 4, stride, 1, bias=not normalize)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(in_channels, 64, normalize=False),
            *block(64, 128),
            *block(128, 256),
            *block(256, 512, stride=1),
            nn.Conv2d(512, 1, 4, 1, 1),
        )

    def forward(self, x):
        return self.model(x)


class EdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def _grad(self, x):
        gx = F.conv2d(x, self.sobel_x, padding=1)
        gy = F.conv2d(x, self.sobel_y, padding=1)
        return torch.sqrt(gx.pow(2) + gy.pow(2) + 1e-6)

    def forward(self, pred, target):
        return F.l1_loss(self._grad(pred), self._grad(target))


def get_stage_loss_weights(stage):
    weights = {
        "l1": 1.0,
        "perceptual": 0.0,
        "gan": 0.0,
        "edge": 0.0,
        "seg": 0.0,
        "det": 0.0,
    }
    if stage >= 2:
        weights["perceptual"] = 0.1
        weights["edge"] = 0.1
    if stage >= 3:
        weights["seg"] = 0.2
        weights["det"] = 0.1
    if stage >= 4:
        weights["gan"] = 0.01
    return weights


def save_visual_comparison(epoch, model, dataloader, device, save_dir, stage):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        samples_saved = 0
        for i, (lr_images, hr_image) in enumerate(dataloader):
            if i not in [10, 30, 50]:
                continue

            lr_imgs = [img.to(device) for img in lr_images]
            hr_img = hr_image.to(device)
            outputs = model(lr_imgs, stage=stage)
            sr_img = outputs["sr"]

            lr_display = lr_imgs[1 if len(lr_imgs) > 1 else 0].squeeze().cpu().numpy()
            sr_display = sr_img.squeeze().cpu().numpy()
            hr_display = hr_img.squeeze().cpu().numpy()

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle(f"Epoch {epoch} Sample {i}", fontsize=14)
            axes[0].imshow(lr_display, cmap="gray")
            axes[0].set_title("LR Input (T2)")
            axes[1].imshow(sr_display, cmap="gray", vmin=0, vmax=1)
            axes[1].set_title("SR Output")
            axes[2].imshow(hr_display, cmap="gray", vmin=0, vmax=1)
            axes[2].set_title("HR Target")
            for ax in axes:
                ax.axis("off")
            plt.savefig(os.path.join(save_dir, f"sample_{samples_saved + 1}.png"))
            plt.close(fig)
            samples_saved += 1
            if samples_saved == 3:
                break
    model.train()


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    # Determine LR input size based on scale
    lr_hw = 64 if args.scale_factor == 4 else 128
    model = SwinMMHCA(n_inputs=args.n_inputs, scale=args.scale_factor, height=lr_hw, width=lr_hw).to(device)
    discriminator = PatchDiscriminator70().to(device)

    optimizer_g = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4, betas=(0.9, 0.999))
    optimizer_d = optim.AdamW(discriminator.parameters(), lr=args.lr * 0.5, weight_decay=1e-4, betas=(0.9, 0.999))

    scaler_g = GradScaler("cuda", enabled=use_amp)
    scaler_d = GradScaler("cuda", enabled=use_amp)

    criterion_l1 = nn.L1Loss()
    criterion_gan = nn.BCEWithLogitsLoss()
    criterion_edge = EdgeLoss().to(device)
    lpips_fn = lpips.LPIPS(net="alex").to(device)

    train_dataset = PreprocessedSuperResDataset(
        args.dataset_root,
        split="train",
        scale_factor=args.scale_factor,
    )
    val_dataset = PreprocessedSuperResDataset(
        args.dataset_root,
        split="validation",
        scale_factor=args.scale_factor,
        shuffle=False,
    )

    persistent_workers = args.num_workers > 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=use_amp,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    os.makedirs("results", exist_ok=True)
    csv_file = os.path.join("results", f"scale_{args.scale_factor}_stage_{args.training_stage}_metrics.csv")
    if not (args.resume and os.path.exists(csv_file)):
        with open(csv_file, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "Epoch",
                    "Stage",
                    "Scale",
                    "Train_Loss_G",
                    "Train_Loss_D",
                    "Val_PSNR",
                    "Val_SSIM",
                    "Val_LPIPS",
                ]
            )

    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            if "discriminator_state_dict" in checkpoint:
                discriminator.load_state_dict(checkpoint["discriminator_state_dict"], strict=False)
            if "optimizer_g_state_dict" in checkpoint:
                optimizer_g.load_state_dict(checkpoint["optimizer_g_state_dict"])
            if "optimizer_d_state_dict" in checkpoint:
                optimizer_d.load_state_dict(checkpoint["optimizer_d_state_dict"])
            if "scaler_g_state_dict" in checkpoint:
                scaler_g.load_state_dict(checkpoint["scaler_g_state_dict"])
            if "scaler_d_state_dict" in checkpoint:
                scaler_d.load_state_dict(checkpoint["scaler_d_state_dict"])
        else:
            model.load_state_dict(checkpoint, strict=False)

    stage_weights = get_stage_loss_weights(args.training_stage)

    for epoch in range(args.start_epoch, args.start_epoch + args.epochs):
        display_epoch = epoch + 1
        model.train()
        discriminator.train()
        epoch_loss_g = 0.0
        epoch_loss_d = 0.0

        for step, (lr_imgs, hr_img) in enumerate(train_loader, start=1):
            lr_imgs = [img.to(device) for img in lr_imgs]
            hr_img = hr_img.to(device)

            if args.training_stage >= 4:
                optimizer_d.zero_grad(set_to_none=True)
                with autocast(device_type=device.type, enabled=use_amp):
                    with torch.no_grad():
                        fake_detached = model(lr_imgs, stage=args.training_stage)["sr"]
                    pred_real = discriminator(hr_img)
                    pred_fake = discriminator(fake_detached.detach())
                    loss_d_real = criterion_gan(pred_real, torch.ones_like(pred_real) * 0.9)
                    loss_d_fake = criterion_gan(pred_fake, torch.zeros_like(pred_fake))
                    loss_d = 0.5 * (loss_d_real + loss_d_fake)
                scaler_d.scale(loss_d).backward()
                scaler_d.step(optimizer_d)
                scaler_d.update()
            else:
                loss_d = torch.zeros((), device=device)

            optimizer_g.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=use_amp):
                outputs = model(lr_imgs, stage=args.training_stage)
                sr_img = outputs["sr"]

                loss_l1 = criterion_l1(sr_img, hr_img)
                loss_perc = lpips_fn(sr_img * 2.0 - 1.0, hr_img * 2.0 - 1.0).mean()
                loss_edge = criterion_edge(sr_img, hr_img)

                aux_losses = model.auxiliary_losses(outputs["seg_logits"], outputs["det_logits"], hr_img)
                loss_seg = aux_losses["seg"]
                loss_det = aux_losses["det"]

                if args.training_stage >= 4:
                    pred_fake = discriminator(sr_img)
                    loss_gan = criterion_gan(pred_fake, torch.ones_like(pred_fake))
                else:
                    loss_gan = torch.zeros((), device=device)

                loss_g = (
                    stage_weights["l1"] * loss_l1
                    + stage_weights["perceptual"] * loss_perc
                    + stage_weights["gan"] * loss_gan
                    + stage_weights["edge"] * loss_edge
                    + stage_weights["seg"] * loss_seg
                    + stage_weights["det"] * loss_det
                )

            scaler_g.scale(loss_g).backward()
            scaler_g.unscale_(optimizer_g)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler_g.step(optimizer_g)
            scaler_g.update()

            epoch_loss_g += loss_g.item()
            epoch_loss_d += loss_d.item()

            if step % 10 == 0:
                print(
                    f"Epoch [{display_epoch}/{args.start_epoch + args.epochs}] "
                    f"Step [{step}/{len(train_loader)}] "
                    f"Stage {args.training_stage} "
                    f"LossG: {loss_g.item():.4f} LossD: {loss_d.item():.4f}"
                )

        avg_loss_g = epoch_loss_g / max(1, len(train_loader))
        avg_loss_d = epoch_loss_d / max(1, len(train_loader))
        print(
            f"Epoch [{display_epoch}/{args.start_epoch + args.epochs}] "
            f"Stage {args.training_stage} "
            f"Avg LossG: {avg_loss_g:.4f} Avg LossD: {avg_loss_d:.4f}"
        )

        if display_epoch % args.val_interval == 0:
            model.eval()
            total_psnr = 0.0
            total_ssim = 0.0
            total_lpips = 0.0

            with torch.no_grad():
                for v_lr_imgs, v_hr_img in val_loader:
                    v_lr_imgs = [img.to(device) for img in v_lr_imgs]
                    v_hr_img = v_hr_img.to(device)
                    outputs = model(v_lr_imgs, stage=args.training_stage)
                    v_sr = outputs["sr"]

                    total_psnr += calculate_psnr(
                        v_sr,
                        v_hr_img,
                        scale=args.scale_factor,
                        metric=psnr_metric,
                    ).item()
                    total_ssim += calculate_ssim(
                        v_sr,
                        v_hr_img,
                        scale=args.scale_factor,
                        metric=ssim_metric,
                    ).item()
                    cropped_sr, cropped_hr = crop_border(v_sr, v_hr_img, scale=args.scale_factor)
                    total_lpips += lpips_fn(cropped_sr * 2.0 - 1.0, cropped_hr * 2.0 - 1.0).mean().item()

            val_len = max(1, len(val_loader))
            avg_psnr = total_psnr / val_len
            avg_ssim = total_ssim / val_len
            avg_lpips = total_lpips / val_len
            print(
                f"Validation Scale {args.scale_factor} Stage {args.training_stage}: "
                f"PSNR={avg_psnr:.4f} SSIM={avg_ssim:.4f} LPIPS={avg_lpips:.4f}"
            )

            with open(csv_file, "a", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [display_epoch, args.training_stage, args.scale_factor, avg_loss_g, avg_loss_d, avg_psnr, avg_ssim, avg_lpips]
                )

            save_visual_comparison(
                display_epoch,
                model,
                val_loader,
                device,
                f"epoch_visuals/scale_{args.scale_factor}_stage_{args.training_stage}_epoch_{display_epoch}",
                args.training_stage,
            )

            os.makedirs("epoch_checkpoints", exist_ok=True)
            checkpoint_path = f"epoch_checkpoints/scale_{args.scale_factor}_stage_{args.training_stage}_epoch_{display_epoch}.pth"
            torch.save(
                {
                    "epoch": display_epoch,
                    "stage": args.training_stage,
                    "scale": args.scale_factor,
                    "model_state_dict": model.state_dict(),
                    "discriminator_state_dict": discriminator.state_dict(),
                    "optimizer_g_state_dict": optimizer_g.state_dict(),
                    "optimizer_d_state_dict": optimizer_d.state_dict(),
                    "scaler_g_state_dict": scaler_g.state_dict(),
                    "scaler_d_state_dict": scaler_d.state_dict(),
                },
                checkpoint_path,
            )
            print(f"Saved checkpoint: {checkpoint_path}")

    if args.save_checkpoint:
        torch.save(
            {
                "epoch": args.start_epoch + args.epochs,
                "stage": args.training_stage,
                "model_state_dict": model.state_dict(),
                "discriminator_state_dict": discriminator.state_dict(),
                "optimizer_g_state_dict": optimizer_g.state_dict(),
                "optimizer_d_state_dict": optimizer_d.state_dict(),
                "scaler_g_state_dict": scaler_g.state_dict(),
                "scaler_d_state_dict": scaler_d.state_dict(),
            },
            args.save_checkpoint,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--dataset_root", type=str, default="processed_data")
    parser.add_argument("--scale_factor", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n_inputs", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--save_checkpoint", type=str, default=None)
    parser.add_argument("--training_stage", type=int, default=1, choices=[1, 2, 3, 4])
    parser.add_argument("--val_interval", type=int, default=5)
    train(parser.parse_args())
