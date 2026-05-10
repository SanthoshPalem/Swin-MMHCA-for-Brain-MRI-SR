import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import importlib.util
import types

# ==========================================
# 1. PATH CONFIGURATION
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MHCA_ROOT = os.path.join(BASE_DIR, "MHCA-main", "MHCA-main", "edsr")
SWIN_SRC = os.path.join(BASE_DIR, "src")
OUTPUT_ROOT = os.path.join(BASE_DIR, "qualitative_results")

# ==========================================
# 2. MODEL LOADING UTILS
# ==========================================
def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

class Args:
    def __init__(self, scale=4, use_mhca_3=True, use_nav=False):
        self.n_resblocks = 16
        self.n_feats = 64
        self.res_scale = 0.1
        self.scale = [scale]
        self.rgb_range = 1.0 
        self.n_colors = 1
        self.ratio = 0.5
        self.shift_mean = False # Medical images typically don't use this shift in MHCA paper
        self.use_mhca_2 = False
        self.use_mhca_3 = use_mhca_3
        self.use_nav = use_nav
        self.use_attention_resblock = True

# Load EDSR+MHCA/MMHCA
mhca_common = load_module("common", os.path.join(MHCA_ROOT, "common.py"))
edsr_mod = load_module("edsr_model", os.path.join(MHCA_ROOT, "edsr.py"))
edsr_nav_mod = load_module("edsr_nav_model", os.path.join(MHCA_ROOT, "edsr_nav.py"))

# Load Swin-MMHCA
models_pkg = types.ModuleType("models")
models_pkg.__path__ = [os.path.join(SWIN_SRC, "models")]
sys.modules["models"] = models_pkg
load_module("models.common", os.path.join(SWIN_SRC, "models", "common.py"))
load_module("models.swin_transformer_v2", os.path.join(SWIN_SRC, "models", "swin_transformer_v2.py"))
swin_mmhca_mod = load_module("models.swin_mmhca", os.path.join(SWIN_SRC, "models", "swin_mmhca.py"))
SwinMMHCA = swin_mmhca_mod.SwinMMHCA

# ==========================================
# 3. UTILITIES
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_img(tensor_np, path):
    # Ensure range [0, 255]
    img_np = np.clip(tensor_np, 0, 1)
    img_np = (img_np * 255.0).astype(np.uint8)
    Image.fromarray(img_np).save(path)

# ==========================================
# 4. INFERENCE & INDIVIDUAL SAVING
# ==========================================
def run_comparison(sample_identifiers, scale=4):
    print(f"\n[+] Generating individual outputs for x{scale}...")
    
    # Init Models
    m_mhca = edsr_mod.EDSR(Args(scale=scale, use_mhca_3=True)).to(DEVICE).eval()
    m_mmhca = edsr_nav_mod.EDSR_Nav(Args(scale=scale, use_mhca_3=True, use_nav=True)).to(DEVICE).eval()
    m_ours = SwinMMHCA(n_inputs=3, scale=scale, height=256//scale, width=256//scale).to(DEVICE).eval()
    
    # Load Weights
    def load_w(m, p):
        if os.path.exists(p):
            sd = torch.load(p, map_location=DEVICE)
            m.load_state_dict(sd if 'state_dict' not in sd else sd['state_dict'], strict=False)
            print(f"Loaded {os.path.basename(p)}")
        else:
            print(f"Warning: Missing weight at {p}")
    
    load_w(m_mhca, os.path.join(MHCA_ROOT, "pretrained_models", f"model_single_input_IXI_x{scale}.pt"))
    load_w(m_mmhca, os.path.join(MHCA_ROOT, "pretrained_models", f"model_multi_input_IXI_x{scale}.pt"))
    
    ours_w_path = os.path.join(BASE_DIR, "epoch_checkpoints", 
                               "stage_1_epoch_250.pth" if scale==4 else "scale_2_stage_1_epoch_120.pth")
    load_w(m_ours, ours_w_path)

    # Setup Folders
    models = ["LR", "Bicubic", "EDSR-MHCA", "EDSR-MMHCA", "Swin-MMHCA", "GT"]
    scale_folder = os.path.join(OUTPUT_ROOT, f"x{scale}")
    for m in models:
        os.makedirs(os.path.join(scale_folder, m), exist_ok=True)

    proc_dir = os.path.join(BASE_DIR, "processed_data")

    for ident in sample_identifiers:
        file_path = os.path.join(proc_dir, ident if ident.endswith('.pt') else ident + ".pt")
        if not os.path.exists(file_path):
            print(f"[-] Warning: {file_path} not found. Skipping.")
            continue
        
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        
        # Load data
        data_tensor = torch.load(file_path, weights_only=True)
        # T1, T2, PD are channels 0, 1, 2. HR target is T2 (channel 1)
        hr_image = data_tensor[1].unsqueeze(0) 
        
        lr_hw = 256 // scale
        lr_images = []
        for m_idx in range(3):
            m_hr = data_tensor[m_idx].unsqueeze(0).unsqueeze(0)
            m_lr = F.interpolate(m_hr, size=(lr_hw, lr_hw), mode='bicubic', align_corners=False)
            lr_images.append(torch.clamp(m_lr.squeeze(0), 0, 1))
        
        # Tensors to Device
        t_lrs = [img.unsqueeze(0).to(DEVICE) for img in lr_images] 
        t_hr = hr_image.unsqueeze(0).to(DEVICE) 
        
        t_l1, t_l2, t_lpd = t_lrs[0], t_lrs[1], t_lrs[2]

        with torch.no_grad():
            # Bicubic
            s_bi = F.interpolate(t_l2, scale_factor=scale, mode='bicubic', align_corners=False).squeeze().cpu().numpy()
            
            # Baseline MHCA (Single Input T2)
            # Try range adjustment if output was near-zero (likely trained on 0-255)
            s_mhca = m_mhca(t_l2).squeeze().cpu().numpy()
            if s_mhca.max() < 0.2: # Heuristic check for invisible output
                print(f"    [!] Detected low-range MHCA output ({s_mhca.max():.4f}). Applying scaling...")
                s_mhca = m_mhca(t_l2 * 255.0).squeeze().cpu().numpy() / 255.0
            
            # Baseline MMHCA (Multi Input T2, PD)
            s_mmhca = m_mmhca([t_l2, t_lpd]).squeeze().cpu().numpy()
            if s_mmhca.max() < 0.2:
                print(f"    [!] Detected low-range MMHCA output ({s_mmhca.max():.4f}). Applying scaling...")
                s_mmhca = m_mmhca([t_l2 * 255.0, t_lpd * 255.0]).squeeze().cpu().numpy() / 255.0
            
            # Ours Swin-MMHCA
            s_ours = m_ours([t_l1, t_l2, t_lpd])['sr'].squeeze().cpu().numpy()

        # Save Individual Images
        save_img(t_l2.squeeze().cpu().numpy(), os.path.join(scale_folder, "LR", f"{base_name}_LR.png"))
        save_img(s_bi, os.path.join(scale_folder, "Bicubic", f"{base_name}_Bicubic.png"))
        save_img(s_mhca, os.path.join(scale_folder, "EDSR-MHCA", f"{base_name}_MHCA.png"))
        save_img(s_mmhca, os.path.join(scale_folder, "EDSR-MMHCA", f"{base_name}_MMHCA.png"))
        save_img(s_ours, os.path.join(scale_folder, "Swin-MMHCA", f"{base_name}_Swin.png"))
        save_img(hr_image.squeeze().numpy(), os.path.join(scale_folder, "GT", f"{base_name}_GT.png"))
        
        print(f"    [OK] Saved results for {base_name}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    default_files = [
        "IXI002-Guys-0828_slice_065.pt",
        "IXI016-Guys-0697_slice_065.pt",
        "IXI017-Guys-0698_slice_065.pt"
    ]
    parser.add_argument('--files', nargs='+', default=default_files, help='Specific .pt filenames')
    args = parser.parse_args()

    # Create individual outputs for both scales
    run_comparison(args.files, scale=4)
    run_comparison(args.files, scale=2)
    
    print(f"\nAll qualitative results saved in: {OUTPUT_ROOT}")
