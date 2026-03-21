import torch
from torchvision.transforms import ToTensor, ToPILImage, Resize, Compose
import os
from PIL import Image, ImageDraw
import argparse
import sys
import numpy as np

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from src.models.swin_mmhca import SwinMMHCA

def run_inference(lr_image_paths, checkpoint_path, output_dir, n_inputs, scale_factor, target_hw=(64, 64)):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Load Model ---
    model = SwinMMHCA(n_inputs=n_inputs, scale=scale_factor, height=target_hw[0], width=target_hw[1]).to(device)
    
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
    else:
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    model.eval()

    # --- Load and Preprocess Input Images ---
    transform_pipeline = Compose([
        Resize(target_hw),
        ToTensor()
    ])
    
    lr_tensors = []
    for path in lr_image_paths:
        try:
            img_pil = Image.open(path).convert('L')
            lr_tensors.append(transform_pipeline(img_pil).unsqueeze(0).to(device))
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return

    if len(lr_tensors) == 1 and n_inputs == 3:
        # Duplicate if only one provided but 3 expected
        model_input = [lr_tensors[0]] * 3
    elif len(lr_tensors) == n_inputs:
        model_input = lr_tensors
    else:
        raise ValueError(f"Expected {n_inputs} inputs, got {len(lr_tensors)}")

    # --- Run Inference ---
    with torch.no_grad():
        sr_img, seg_mask_logits, bboxes = model(model_input)
        seg_mask = torch.sigmoid(seg_mask_logits)
        
    print(f"SR Output - Min: {sr_img.min():.4f}, Max: {sr_img.max():.4f}, Mean: {sr_img.mean():.4f}")
    print(f"Seg Mask - Min: {seg_mask.min():.4f}, Max: {seg_mask.max():.4f}, Mean: {seg_mask.mean():.4f}")
    print(f"BBoxes Conf - Max: {bboxes[..., 4].max():.4f}")

    # --- Post-process and Save Outputs ---
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    to_pil = ToPILImage()

    # 1. Save SR Image
    # If values are extremely low, normalize for visualization
    if sr_img.max() < 0.01:
        print("Warning: SR output values are very low. Normalizing for visualization.")
        sr_img_viz = (sr_img - sr_img.min()) / (sr_img.max() - sr_img.min() + 1e-8)
    else:
        sr_img_viz = torch.clamp(sr_img, 0, 1)
        
    sr_pil = to_pil(sr_img_viz.squeeze(0).cpu())
    sr_path = os.path.join(output_dir, "sr_reconstruction.png")
    sr_pil.save(sr_path)

    # 2. Save Segmentation Mask
    if seg_mask.max() < 0.01:
        print("Warning: Seg mask values are very low. Normalizing for visualization.")
        seg_mask_viz = (seg_mask - seg_mask.min()) / (seg_mask.max() - seg_mask.min() + 1e-8)
    else:
        seg_mask_viz = torch.clamp(seg_mask, 0, 1)
        
    seg_pil = to_pil(seg_mask_viz.squeeze(0).cpu())
    seg_path = os.path.join(output_dir, "segmentation_mask.png")
    seg_pil.save(seg_path)

    # 3. Save Image with Bounding Boxes
    # Use the SR image as the background for the bounding boxes
    draw_img = sr_pil.convert("RGB")
    draw = ImageDraw.Draw(draw_img)
    
    # bboxes shape: [B, 5, 5] -> [1, 5, 5]
    # Each box: [x, y, w, h, conf]
    boxes = bboxes.squeeze(0).cpu().numpy()
    img_w, img_h = draw_img.size

    found_boxes = 0
    for box in boxes:
        x, y, w, h, conf = box
        if conf > 0.3: # Lowered threshold slightly for visualization
            found_boxes += 1
            # Scale coordinates back to image size
            left = (x - w/2) * img_w
            top = (y - h/2) * img_h
            right = (x + w/2) * img_w
            bottom = (y + h/2) * img_h
            draw.rectangle([left, top, right, bottom], outline="red", width=2)
            draw.text((left, top), f"{conf:.2f}", fill="red")
    
    print(f"Found {found_boxes} boxes with confidence > 0.3")

    det_path = os.path.join(output_dir, "detection_result.png")
    draw_img.save(det_path)

    det_path = os.path.join(output_dir, "detection_result.png")
    draw_img.save(det_path)

    print(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference for Dual-Path SwinMMHCA')
    parser.add_argument('--inputs', type=str, nargs='+', required=True, help='Paths to input images (T1, T2, PD)')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--outdir', type=str, default='inference_results', help='Output directory')
    parser.add_argument('--scale', type=int, default=4, help='Scale factor')

    args = parser.parse_args()

    run_inference(args.inputs, args.checkpoint, args.outdir, len(args.inputs), args.scale)
