import subprocess
import re
import csv
import os

def run_evaluation():
    epochs = [10, 20, 30, 40, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    results_file = 'evaluation_results.csv'
    checkpoint_dir = 'epoch_checkpoints'
    
    # Header for the CSV file
    with open(results_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'PSNR', 'SSIM', 'LPIPS'])

    for epoch in epochs:
        # Determine the checkpoint filename. Based on 'dir' output, it's 'gan_epoch_X.pth'.
        checkpoint_path = os.path.join(checkpoint_dir, f'gan_epoch_{epoch}.pth')
        
        if not os.path.exists(checkpoint_path):
            print(f"Skipping epoch {epoch}: Checkpoint not found at {checkpoint_path}")
            continue
            
        print(f"Evaluating epoch {epoch}...")
        
        # Build the command to call evaluate.py
        command = [
            'python', 'evaluate.py',
            '--checkpoint_path', checkpoint_path,
            '--batch_size', '1',
            '--num_workers', '0'
        ]
        
        try:
            # Execute evaluate.py and capture the output
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                print(f"Error evaluating epoch {epoch}:\n{stderr}")
                continue
            
            # Extract metrics using regex
            psnr_match = re.search(r'Average PSNR:\s+([\d.]+)', stdout)
            ssim_match = re.search(r'Average SSIM:\s+([\d.]+)', stdout)
            lpips_match = re.search(r'Average LPIPS:\s+([\d.]+)', stdout)
            
            if psnr_match and ssim_match and lpips_match:
                psnr = float(psnr_match.group(1))
                ssim = float(ssim_match.group(1))
                lpips = float(lpips_match.group(1))
                
                print(f"Epoch {epoch}: PSNR={psnr:.4f}, SSIM={ssim:.4f}, LPIPS={lpips:.4f}")
                
                # Append to CSV
                with open(results_file, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, psnr, ssim, lpips])
            else:
                print(f"Could not parse metrics for epoch {epoch}. Output was:\n{stdout}")
                
        except Exception as e:
            print(f"An unexpected error occurred for epoch {epoch}: {str(e)}")

if __name__ == '__main__':
    run_evaluation()
