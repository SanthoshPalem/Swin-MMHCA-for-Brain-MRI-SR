# Swin-MMHCA Training Progress & Strategy (Target: 32.9 dB for Scale 4)

## Scale 4 Evaluation Results (Stage 1)
The model has achieved the updated performance target:

| Metric | Scale 4 Result |
| :--- | :--- |
| **Val PSNR** | **32.90 dB** |

## Scale 2 Final Evaluation (Epoch 120)
The model has achieved exceptional performance on the Scale 2 task:

| Metric | Epoch 120 (Scale 2) |
| :--- | :--- |
| **Test PSNR** | **40.4850 dB** |
| **Test SSIM** | **0.9864** |
| **Test LPIPS** | **0.0107** |

## Scale 4 Current Progress Analysis (Stage 1)
Based on the latest logs from Epoch 55 to 90, the model is showing a healthy upward trend in image fidelity:

| Metric | Epoch 55 | Epoch 90 | Improvement |
| :--- | :--- | :--- | :--- |
| **Val_PSNR** | 31.1593 | **31.5467** | **+0.3874 dB** |
| **Val_SSIM** | 0.9363 | **0.9394** | **+0.0031** |
| **Train_Loss_G** | 0.0113 | **0.0108** | **-0.0005** |

### Key Observations:
* **Steady Growth:** PSNR is increasing at a rate of approximately **0.05 - 0.06 dB every 5 epochs**.
* **High Structural Fidelity:** An SSIM of **0.9394** indicates the model has already mastered the brain's global structure.
* **No Plateau:** `Train_Loss_G` is still decreasing, meaning the model hasn't hit its "ceiling" yet.

---

## Strategy to Reach 32.7 dB

To reach the target of **32.7 dB** (an additional ~1.15 dB), you need to continue Stage 1 training. This stage focuses purely on $L_1$ pixel-wise accuracy, which is the most effective for maximizing PSNR.

### Fine-Tuning Phase (Epoch 250+)
* **Recommended Learning Rate:** `5e-5`
* **Objective:** Stabilize weight adjustments to maximize final PSNR gains beyond 32.9 dB.

### Recommended Command
Run this command to resume training from epoch 250:
```powershell
python run.py --resume epoch_checkpoints/stage_1_epoch_250.pth --start_epoch 250 --epochs 50 --lr 5e-5 --training_stage 1 --val_interval 10 --batch_size 16
```

### Technical Rationale:
1. **Lower Learning Rate (`8e-5`):** By reducing the LR slightly from the standard `1e-4`, the optimizer can make more precise weight adjustments. This "fine-tuning" mode is essential for closing the 1.15 dB gap without the loss fluctuating.
2. **Extended Epochs:** Given the current growth rate, hitting 32.7 dB will likely take another 150 epochs.
3. **Val Interval:** Set to `10` to reduce the time spent on validation, allowing the GPU to focus on training.

---

## Monitoring and Adjustments
* **Check Progress:** Every 10-20 epochs, check `results/stage_1_training_metrics.csv`.
* **Plateau Detection:** If the PSNR stops increasing for more than 30 epochs (e.g., stays at 32.2 dB), stop the training and restart with a much lower learning rate: `--lr 1e-5`.
* **GPU Memory:** The **NVIDIA DGX A100** provides ample VRAM (40GB/80GB), allowing for larger batch sizes if needed.

---
**Status:** Ready to Resume.
