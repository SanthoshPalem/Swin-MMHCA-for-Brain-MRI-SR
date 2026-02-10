import torch
from torchvision.transforms import ToTensor, ToPILImage, Resize, Compose
import os
from PIL import Image
import argparse
import sys

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from src.models.swin_mmhca import SwinMMHCA

def run_inference(lr_image_path, checkpoint_path, output_path, n_inputs, scale_factor, target_hw=(64, 64)): # Added target_hw
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Load Model ---
    # The model expects its internal height and width for positional encoding etc.
    # These are hardcoded in SwinMMHCA's __init__ to 64, 64.
    model = SwinMMHCA(n_inputs=n_inputs, scale=scale_factor, height=target_hw[0], width=target_hw[1]).to(device)
    
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    model.eval() # Set model to evaluation mode

    # --- Load and Preprocess Input Image ---
    # Add Resize to the transformation pipeline
    transform_pipeline = Compose([
        Resize(target_hw), # Resize the image to match model's expected input dimensions
        ToTensor()
    ])
    
    try:
        lr_image_pil = Image.open(lr_image_path).convert('L') # Convert to grayscale
    except FileNotFoundError:
        print(f"Error: Low-resolution image not found at {lr_image_path}")
        return
    
    lr_image_tensor_single = transform_pipeline(lr_image_pil).unsqueeze(0).to(device) # Add batch dimension

    # --- Prepare input for the model ---
    if n_inputs == 1:
        model_input = lr_image_tensor_single
    elif n_inputs == 3:
        # If model expects 3 inputs but only one path is given, duplicate the input
        # This assumes the user wants to feed the same image to all 3 input branches
        model_input = [lr_image_tensor_single, lr_image_tensor_single, lr_image_tensor_single]
    else:
        raise ValueError(f"Model configured for {n_inputs} inputs, but script expects 1 or 3.")

    # --- Run Inference ---
    with torch.no_grad():
        sr_output_tensor = model(model_input)

    # --- Post-process and Save Output ---
    sr_output_tensor = torch.clamp(sr_output_tensor, 0, 1) # Clamp values to [0, 1]
    
    to_pil_image = ToPILImage()
    sr_image_pil = to_pil_image(sr_output_tensor.squeeze(0).cpu()) # Remove batch dim, move to CPU
    
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sr_image_pil.save(output_path)
    print(f"Super-resolved image saved to: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform inference with SwinMMHCA model on a single JPEG image.')
    parser.add_argument('--lr_image_path', type=str, required=True, help='Path to the low-resolution JPEG image.')
    parser.add_argument('--checkpoint_path', type=str, default='epoch_checkpoints/swin_mmhca_x4_epoch_100.pth',
                        help='Path to the model checkpoint (.pth file).')
    parser.add_argument('--output_path', type=str, default='super_resolved_output.png',
                        help='Path to save the super-resolved PNG image.')
    parser.add_argument('--n_inputs', type=int, default=3,
                        help='Number of input modalities (1 for single modality model, 3 for multi-modal model).')
    parser.add_argument('--scale_factor', type=int, default=4, help='Super-resolution scale factor.')
    parser.add_argument('--target_height', type=int, default=64, help='Target height for input image resizing.')
    parser.add_argument('--target_width', type=int, default=64, help='Target width for input image resizing.')


    args = parser.parse_args()

    run_inference(args.lr_image_path, args.checkpoint_path, args.output_path, args.n_inputs, args.scale_factor,
                  target_hw=(args.target_height, args.target_width))
