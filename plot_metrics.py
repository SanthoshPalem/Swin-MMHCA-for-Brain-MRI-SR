import pandas as pd
import matplotlib.pyplot as plt
import os

def generate_plots(csv_file='evaluation_results.csv'):
    if not os.path.exists(csv_file):
        print(f"Error: Results file '{csv_file}' not found.")
        return

    # Load results
    df = pd.read_csv(csv_file)
    df.sort_values('Epoch', inplace=True)
    
    # Define groups
    # Group 1: Epochs 10, 20, 30, 40, 50
    early_df = df[df['Epoch'] <= 50]
    
    # Group 2: Epochs 50, 100, 150, 200, 250, 300, 350, 400, 450, 500
    # Note: Epoch 50 is included in both for continuity if desired,
    # though the prompt specified Group 2: Epochs 50-500 (step 50).
    late_df = df[df['Epoch'] >= 50]
    
    # --- Plotting Early Training ---
    plt.figure(figsize=(10, 6))
    
    # Use secondary Y-axis for LPIPS and SSIM if scale difference is too high
    ax1 = plt.gca()
    ax1.plot(early_df['Epoch'], early_df['PSNR'], marker='o', label='PSNR (dB)', color='tab:blue')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('PSNR', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(early_df['Epoch'], early_df['SSIM'], marker='s', label='SSIM', color='tab:red')
    ax2.plot(early_df['Epoch'], early_df['LPIPS'], marker='^', label='LPIPS', color='tab:green')
    ax2.set_ylabel('SSIM / LPIPS', color='black')
    ax2.tick_params(axis='y', labelcolor='black')

    plt.title('Early Training Metrics (Epochs 10-50)')
    
    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.tight_layout()
    plt.savefig('early_training_plot.png')
    print("Saved early_training_plot.png")
    plt.close()

    # --- Plotting Late Training ---
    plt.figure(figsize=(10, 6))
    
    ax1 = plt.gca()
    ax1.plot(late_df['Epoch'], late_df['PSNR'], marker='o', label='PSNR (dB)', color='tab:blue')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('PSNR', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(late_df['Epoch'], late_df['SSIM'], marker='s', label='SSIM', color='tab:red')
    ax2.plot(late_df['Epoch'], late_df['LPIPS'], marker='^', label='LPIPS', color='tab:green')
    ax2.set_ylabel('SSIM / LPIPS', color='black')
    ax2.tick_params(axis='y', labelcolor='black')

    plt.title('Later Training Metrics (Epochs 50-500)')
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.tight_layout()
    plt.savefig('late_training_plot.png')
    print("Saved late_training_plot.png")
    plt.close()

if __name__ == '__main__':
    generate_plots()
