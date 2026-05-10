import os
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import random

def main():
    parser = argparse.ArgumentParser(description="Preview .pt tensor slices")
    # Accept multiple files via space seperation
    parser.add_argument('--files', nargs='+', default=None, help='Specific filenames to preview, separated by space')
    # Default to picking 10 randomly
    parser.add_argument('--count', type=int, default=10, help='If no files specified, how many random slices to pick')
    args = parser.parse_args()

    project_root = Path(__file__).parent.absolute()
    processed_dir = project_root / "processed_data"
    
    pt_file_paths = []
    
    if args.files:
        for f in args.files:
            p = project_root / f
            if not p.exists():
                p = processed_dir / f
            pt_file_paths.append(p)
    else:
        # Pick 10 fully randomized files from processed_data every single run!
        all_pt_files = list(processed_dir.glob("*.pt"))
        if not all_pt_files:
            print(f"Error: No .pt files found in {processed_dir}")
            return
        
        take_count = min(args.count, len(all_pt_files))
        pt_file_paths = random.sample(all_pt_files, take_count)
        
    output_dir = project_root / "preview_images"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    processed_count = 0
    for pt_file_path in pt_file_paths:
        if not pt_file_path.exists():
            print(f"[-] Error: Could not find '{pt_file_path.name}'. Skipping...")
            continue
            
        print(f"[+] Loading native tensor from {pt_file_path.name}...")
        # Load the 3D tensor -> [3, 256, 256]
        tensor = torch.load(pt_file_path, weights_only=True)
        
        base_name = pt_file_path.stem
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(tensor[0].numpy(), cmap='gray')
        axes[0].set_title(f'T1 Modality \n{base_name}')
        axes[0].axis('off')
        
        axes[1].imshow(tensor[1].numpy(), cmap='gray')
        axes[1].set_title(f'T2 Modality \n{base_name}')
        axes[1].axis('off')
        
        axes[2].imshow(tensor[2].numpy(), cmap='gray')
        axes[2].set_title(f'PD Modality \n{base_name}')
        axes[2].axis('off')
        
        save_path = output_dir / f"{base_name}_preview.png"
        plt.tight_layout()
        plt.savefig(str(save_path))
        plt.close(fig) # Prevent memory leaks when looping
        processed_count += 1
        
    print(f"\nSuccess! Rendered {processed_count} PyTorch Visualizations perfectly to '{output_dir.name}/'")

if __name__ == "__main__":
    main()
