import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, ToPILImage, ToTensor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from src.models.swin_mmhca import SwinMMHCA


def run_inference(lr_image_paths, checkpoint_path, output_dir, n_inputs, scale_factor, target_hw=(64, 64)):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SwinMMHCA(n_inputs=n_inputs, scale=scale_factor).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        stage = checkpoint.get("stage", 4)
    else:
        model.load_state_dict(checkpoint, strict=False)
        stage = 4
    model.eval()

    preprocess = Compose([Resize(target_hw), ToTensor()])

    lr_tensors = []
    for path in lr_image_paths:
        image = Image.open(path).convert("L")
        lr_tensors.append(preprocess(image).unsqueeze(0).to(device))

    if len(lr_tensors) == 1 and n_inputs == 3:
        model_input = [lr_tensors[0], lr_tensors[0], lr_tensors[0]]
    elif len(lr_tensors) == n_inputs:
        model_input = lr_tensors
    else:
        raise ValueError(f"Expected {n_inputs} inputs, got {len(lr_tensors)}")

    with torch.no_grad():
        outputs = model(model_input, stage=stage)
        sr_img = outputs["sr"]
        seg_mask = torch.sigmoid(outputs["seg_logits"])
        det_grid = torch.sigmoid(outputs["det_logits"])

    os.makedirs(output_dir, exist_ok=True)
    to_pil = ToPILImage()

    to_pil(sr_img.squeeze(0).cpu()).save(os.path.join(output_dir, "sr_reconstruction.png"))
    to_pil(seg_mask.squeeze(0).cpu()).save(os.path.join(output_dir, "segmentation_mask.png"))

    det_np = det_grid.squeeze().cpu().numpy()
    det_vis = Image.fromarray(np.uint8(np.clip(det_np, 0.0, 1.0) * 255.0), mode="L").resize((256, 256), Image.NEAREST)
    det_vis.save(os.path.join(output_dir, "detection_grid.png"))

    print(f"Saved inference outputs to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference for Swin-MMHCA")
    parser.add_argument("--inputs", type=str, nargs="+", required=True, help="Paths to input images (T1, T2, PD)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--outdir", type=str, default="inference_results", help="Output directory")
    parser.add_argument("--scale", type=int, default=4, help="Scale factor")
    args = parser.parse_args()

    run_inference(args.inputs, args.checkpoint, args.outdir, len(args.inputs), args.scale)
