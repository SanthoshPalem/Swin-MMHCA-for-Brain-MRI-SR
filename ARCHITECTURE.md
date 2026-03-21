# Swin-MMHCA: Edge-Guided GAN for Medical Image Super-Resolution

This document describes the final architecture, components, and performance of the enhanced **Swin-MMHCA** model.

## 1. Architectural Overview
Swin-MMHCA is a multi-task generative adversarial network (GAN) designed for high-fidelity Medical Image Super-Resolution (MISR). It employs a **Pathology-Aware Dual-Path Architecture** that integrates high-resolution Transformer tokenization, edge-guided learning, and adversarial training.

### **Path 1: Super-Resolution (SR) Path**
*   **Swin Transformer V2 Backbone:** 
    *   **High-Resolution Tokenization:** Uses a `patch_size=2` to create a dense 32x32 token grid from a 64x64 input, essential for capturing small tumor features.
    *   **Configuration:** 4 stages with depths `[2, 2, 6, 2]`, number of heads `[3, 6, 12, 24]`, and `embed_dim=96`.
    *   **2D Positional Encoding:** Sinusoidal-based 2D positional embeddings are added before the Transformer stages to preserve spatial relationships.
    *   **Reduced Spatial Compression:** Unlike standard models that compress to 4x4, this model maintains a latent resolution of **8x8**, preventing the loss of high-frequency anatomical details.
*   **Global Residual Learning:** Performs bicubic upsampling on the input (T2) to create a base image; the network learns to predict the "residual" high-frequency textures to add back to this base.

### **Path 2: Medical Context & Edge Path**
*   **Edge Extraction Module:** Uses Sobel operators to extract multi-modal gradient maps (T1, T2, PD). These are processed through a texture-learning branch (CNN) to extract high-frequency features.
*   **MHCA Module:** Multi-Head Cross-Attention identifies critical anatomical regions and integrates learned edge features into the context guidance.
*   **Multitask Heads:**
    *   **Segmentation Head:** A dedicated decoder branch provides anatomical constraints by predicting a binary mask (thresholded at 0.5).
    *   **Detection Head:** A 5x5 grid output for potential lesion localization, encouraging the model to focus on pathologically relevant features.

### **Fusion & Reconstruction**
*   **Cross-Attention Fusion:** SR features dynamically attend to the Edge and Context features via a `CrossAttentionFusion` module, allowing for pathology-aware reconstruction where high-frequency details are "guided" into the final image.
*   **U-Net Style Decoder:** A deep progressive decoder using **PixelShuffle** for upsampling and **Residual Blocks** for refinement. It uses dense skip connections from every stage of the Swin Transformer (64x64, 32x32, 16x16, 8x8).
*   **PatchGAN Discriminator:** A 70x70 PatchGAN discriminator penalizes blurry outputs and forces the generator to produce sharp, perceptually natural textures.

---

## 2. Model Specifications & Training
### **Hyperparameters**
| Parameter | Value |
| :--- | :--- |
| **Input size** | 64 x 64 (LR) |
| **Output size** | 256 x 256 (HR) |
| **Scale Factor** | 4x |
| **Latent Bottleneck** | 8 x 8 |
| **Swin Embed Dim** | 96 |
| **Swin Depths** | [2, 2, 6, 2] |
| **Swin Heads** | [3, 6, 12, 24] |
| **Loss Function** | $10 \cdot L_1 + 1 \cdot L_{perc} (LPIPS) + 2 \cdot L_{edge} + \lambda_{adv} \cdot L_{GAN} + 0.1 \cdot L_{seg}$ |

### **Training Strategy**
*   **Optimizer:** Adam ($\beta_1=0.5, \beta_2=0.999$) with learning rate $1 \times 10^{-4}$.
*   **Mixed Precision:** Utilizes Automatic Mixed Precision (AMP) for faster training and reduced memory footprint.
*   **Adversarial Warming:** Introduction of GAN loss ($\lambda_{adv}=0.1$) is delayed until Epoch 10 to allow the generator to stabilize its reconstruction first.

---

## 3. Data Pipeline
*   **Dataset:** IXI Dataset (T1, T2, PD modalities).
*   **Target:** T2 modality is the primary target for super-resolution.
*   **Preprocessing:** 
    *   Central slice extraction from NIfTI volumes.
    *   Normalization to range [0, 1].
    *   **Multi-modal Alignment:** Uses identical random crop coordinates (256x256 HR) across all three modalities to ensure perfect spatial correspondence.
    *   **Augmentation:** Random cropping and bicubic downsampling (4x) to generate LR-HR pairs.

---

## 4. Performance Metrics (IXI Test Dataset)
Evaluated on the IXI test set (10% hold-out) using the corrected `torchmetrics` (data_range=1.0) and the final `swin_mmhca_final.pth` checkpoint.

| Metric | Score |
| :--- | :--- |
| **PSNR** | 32.5483 dB |
| **SSIM** | 0.9382 |
| **LPIPS** | 0.02 |

---

## 5. Usage
### **Training (GAN + Edge Guided)**
```bash
python run.py --mode train --epochs 500 --batch_size 32 --num_workers 16 --n_inputs 3 --scale_factor 4
```

### **Evaluation**
```bash
python evaluate.py --checkpoint_path pretrained_models/swin_mmhca_final.pth --n_inputs 3 --scale_factor 4
```
