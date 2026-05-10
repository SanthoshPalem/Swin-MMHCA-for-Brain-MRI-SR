import torch
import os
import random
import matplotlib.pyplot as plt
import numpy as np

def visualize_samples(data_dir, num_samples=5):
    # 1. Get list of all .pt files
    files = [f for f in os.listdir(data_dir) if f.endswith('.pt')]
    if not files:
        print("No .pt files found in the directory.")
        return

    # 2. Select random samples
    samples = random.sample(files, min(num_samples, len(files)))
    
    # 3. Create plot grid: num_samples rows x 3 columns (T1, T2, PD)
    fig, axes = plt.subplots(len(samples), 3, figsize=(15, 3 * len(samples)))
    plt.subplots_adjust(wspace=0.1, hspace=0.3)

    modalities = ['T1', 'T2', 'PD']

    for i, filename in enumerate(samples):
        file_path = os.path.join(data_dir, filename)
        # Load the 3-channel tensor [3, 256, 256]
        data = torch.load(file_path)
        
        # Plot each channel
        for j in range(3):
            ax = axes[i, j] if len(samples) > 1 else axes[j]
            slice_2d = data[j].numpy()
            
            ax.imshow(slice_2d, cmap='gray')
            if i == 0:
                ax.set_title(modalities[j], fontsize=14, fontweight='bold')
            if j == 0:
                ax.set_ylabel(filename.split('_slice')[0], fontsize=10)
            
            ax.axis('off')

    # 4. Save the visualization
    output_path = "preprocessing_preview.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    print(f"Visualization saved to {output_path}")

if __name__ == "__main__":
    visualize_samples("processed_data", num_samples=10)
