import os
import torch
import numpy as np
from src.data.preprocess import MRIPreProcessor
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import multiprocessing

# IMPORTANT: Fix multiprocessing issues with PyTorch tensors on DGX servers
try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass

def process_single_subject(subject_data, output_dir):
    base_id, t1_path, t2_path, pd_path = subject_data
    
    # Check if already processed (safely allow resumption)
    if (output_dir / f"{base_id}_slice_000.pt").exists():
        return f"[SKIP] {base_id}: Already processed."

    # Initialize locally in worker to avoid serialization errors
    processor = MRIPreProcessor(threshold=0.01, target_shape=(256, 256, 128))
    
    try:
        subject_slices = processor.process_subject(str(t1_path), str(t2_path), str(pd_path))
        
        if len(subject_slices) < 20:
             return f"[SKIP] {base_id}: Too few valid slices found."
             
        for i, slice_data in enumerate(subject_slices):
            save_path = output_dir / f"{base_id}_slice_{i:03d}.pt"
            # Store as FloatTensor for PyTorch consumption
            torch.save(torch.from_numpy(slice_data).float(), save_path)
            
        return f"[+] {base_id}: Saved {len(subject_slices)} slices."
    except Exception as e:
        return f"[-] {base_id}: Error: {str(e)}"

def preprocess_entire_dataset_dgx():
    project_root = Path(__file__).parent.absolute()
    dataset_root = project_root / "datasets"
    output_dir = project_root / "processed_data"
    
    t1_dir = dataset_root / "IXI-T1"
    t2_dir = dataset_root / "IXI-T2"
    pd_dir = dataset_root / "IXI-PD"
    
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    
    t1_files = sorted(list(t1_dir.glob("*-T1.nii.gz")))
    if not t1_files:
        print(f"No T1 files found in {t1_dir}!")
        return

    valid_subjects = []
    for t1_path in t1_files:
        base_id = t1_path.name.replace("-T1.nii.gz", "")
        t2_path = t2_dir / f"{base_id}-T2.nii.gz"
        pd_path = pd_dir / f"{base_id}-PD.nii.gz"
        
        if t2_path.exists() and pd_path.exists():
            valid_subjects.append((base_id, t1_path, t2_path, pd_path))
            
    print(f"Found {len(valid_subjects)} valid subjects with all modalities.")
    
    # -----------------------------
    # DGX MAX PERFORMANCE CONFIG
    # -----------------------------
    # DGX servers have massive CPU cores. We use 75% max to let the OS breathe
    num_cores = max(1, int(os.cpu_count() * 0.75))
    print(f"Launching HIGH-SPEED MULTIPROCESSING with {num_cores} parallel workers...")
    
    completed = 0
    # Process Pool for true parallelism
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        worker_func = partial(process_single_subject, output_dir=output_dir)
        futures = {executor.submit(worker_func, sub): sub for sub in valid_subjects}
        
        for future in as_completed(futures):
            result = future.result()
            completed += 1
            print(f"[{completed}/{len(valid_subjects)}] {result}")
            
    print(f"\nPreprocessing complete! Data stored in: {output_dir}")

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 DGX High-Speed Multi-Modal Preprocessing Pipeline 🚀")
    print("=" * 60)
    preprocess_entire_dataset_dgx()
