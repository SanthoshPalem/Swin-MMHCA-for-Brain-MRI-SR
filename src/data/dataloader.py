import os
import torch
import torch.utils.data as data
import nibabel as nib
import numpy as np
from PIL import Image
import glob

class MultiModalSuperResDataset(data.Dataset):
    def __init__(self, dataset_root, modalities=['T1', 'T2', 'PD'], scale_factor=4, transform=None, split='train', shuffle=True):
        super(MultiModalSuperResDataset, self).__init__()
        
        self.dataset_root = dataset_root
        self.modalities = modalities
        self.scale_factor = scale_factor
        self.transform = transform
        
        all_samples = self._scan_dataset()
        if shuffle:
            np.random.shuffle(all_samples)
            
        # Percentage-based split: 80% train, 10% validation, 10% test
        num_samples = len(all_samples)
        train_end = int(num_samples * 0.8)
        val_end = int(num_samples * 0.9)
        
        if split == 'train':
            self.samples = all_samples[:train_end]
        elif split == 'validation':
            self.samples = all_samples[train_end:val_end]
        elif split == 'test':
            self.samples = all_samples[val_end:]
        else:
            raise ValueError(f"Invalid split '{split}'. Choose from 'train', 'validation', 'test'.")

    def _scan_dataset(self):
        samples = []
        modality_roots = {mod: os.path.join(self.dataset_root, f'IXI-{mod}') for mod in self.modalities}
        
        primary_modality = self.modalities[0]
        primary_modality_root = modality_roots[primary_modality]
        if not os.path.isdir(primary_modality_root):
            raise FileNotFoundError(f"Directory not found: {primary_modality_root}")

        for filename in os.listdir(primary_modality_root):
            if filename.endswith(f'-{primary_modality}.nii.gz'):
                base_name = filename.replace(f'-{primary_modality}.nii.gz', '')
                
                sample_files = {}
                all_files_exist = True
                for mod in self.modalities:
                    expected_filename = f"{base_name}-{mod}.nii.gz"
                    file_path = os.path.join(modality_roots[mod], expected_filename)
                    if os.path.exists(file_path):
                        sample_files[mod] = file_path
                    else:
                        all_files_exist = False
                        break
                
                if all_files_exist:
                    samples.append(sample_files)
        
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample_files = self.samples[index]
        
        lr_images = []
        hr_image = None
        
        # The target modality for super-resolution is T2
        target_modality = 'T2'
        
        # First, load all original slices for cropping
        original_slices = {}
        for mod in self.modalities:
            file_path = sample_files[mod]
            nii_img = nib.load(file_path)
            img_data = nii_img.get_fdata()
            central_slice_idx = img_data.shape[2] // 2
            slice_data = img_data[:, :, central_slice_idx]
            slice_data = (slice_data - slice_data.min()) / (slice_data.max() - slice_data.min())
            slice_data = (slice_data * 255).astype(np.uint8)
            img_pil = Image.fromarray(slice_data).convert('L') # Grayscale
            original_slices[mod] = img_pil

        # Determine HR patch size and random crop coordinates
        hr_patch_size = 64 * self.scale_factor # e.g., 256 for scale_factor=4
        
        original_width, original_height = original_slices[target_modality].size 
        
        if original_width < hr_patch_size or original_height < hr_patch_size:
            # Fallback: If original image is smaller than target HR patch,
            # resize the whole original image to hr_patch_size and then downsample.
            hr_cropped_pil = original_slices[target_modality].resize((hr_patch_size, hr_patch_size), Image.BICUBIC)
            lr_cropped_pil_target = hr_cropped_pil.resize((64, 64), Image.BICUBIC)

            for mod in self.modalities:
                if mod == target_modality:
                    lr_images.append(self.transform(lr_cropped_pil_target))
                    hr_image = self.transform(hr_cropped_pil)
                else:
                    resized_lr_mod = original_slices[mod].resize((64, 64), Image.BICUBIC)
                    lr_images.append(self.transform(resized_lr_mod))
        else:
            # Randomly crop for images larger than or equal to hr_patch_size
            left = np.random.randint(0, original_width - hr_patch_size + 1)
            top = np.random.randint(0, original_height - hr_patch_size + 1)
            # right = left + hr_patch_size # Not explicitly needed for crop()
            # bottom = top + hr_patch_size # Not explicitly needed for crop()

            # Process each modality using the same crop coordinates
            for mod in self.modalities:
                img_pil = original_slices[mod]
                
                # Crop HR patch
                hr_cropped_pil = img_pil.crop((left, top, left + hr_patch_size, top + hr_patch_size))
                
                # Generate LR patch by downsampling the HR patch
                lr_cropped_pil = hr_cropped_pil.resize((hr_patch_size // self.scale_factor, hr_patch_size // self.scale_factor), Image.BICUBIC)
                
                if self.transform:
                    lr_images.append(self.transform(lr_cropped_pil))
                    if mod == target_modality:
                        hr_image = self.transform(hr_cropped_pil)
        
        return lr_images, hr_image

class PreprocessedSuperResDataset(data.Dataset):
    def __init__(self, processed_dir, modalities=['T1', 'T2', 'PD'], scale_factor=4, split='train', shuffle=True, transform=None):
        super(PreprocessedSuperResDataset, self).__init__()
        self.processed_dir = processed_dir
        self.modalities = modalities
        self.scale_factor = scale_factor
        self.split = split  # Used to mathematically verify Data Augmentation safety
        self.transform = transform # Kept for API compatibility
        
        all_samples = sorted(glob.glob(os.path.join(self.processed_dir, "*.pt")))
        if not all_samples:
            print(f"Warning: No .pt files found in {self.processed_dir}")
            
        subject_dict = {}
        for f in all_samples:
            base_id = os.path.basename(f).split('_slice_')[0]
            if base_id not in subject_dict:
                subject_dict[base_id] = []
            subject_dict[base_id].append(f)
            
        subject_ids = sorted(list(subject_dict.keys()))
        if shuffle:
            np.random.seed(42)  # Deterministic split to prevent data leakage
            np.random.shuffle(subject_ids)
            
        num_subjects = len(subject_ids)
        train_end = int(num_subjects * 0.8)
        val_end = int(num_subjects * 0.9)
        
        if split == 'train':
            split_ids = subject_ids[:train_end]
        elif split == 'validation':
            split_ids = subject_ids[train_end:val_end]
        elif split == 'test':
            split_ids = subject_ids[val_end:]
        else:
            raise ValueError("Invalid split")
            
        self.samples = []
        for sid in split_ids:
            self.samples.extend(subject_dict[sid])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        file_path = self.samples[index]
        data_tensor = torch.load(file_path, weights_only=True) # [3, 256, 256]
        
        # --- DYNAMIC DATA AUGMENTATION (ONLY FOR TRAINING SET) ---
        if self.split == 'train':
            # 50% chance Horizontal Flip
            if torch.rand(1).item() > 0.5:
                data_tensor = torch.flip(data_tensor, dims=[2])
                
            # 50% chance Vertical Flip
            if torch.rand(1).item() > 0.5:
                data_tensor = torch.flip(data_tensor, dims=[1])
                
            # Random 90, 180, 270 Degree Rotations
            k = int(torch.randint(0, 4, (1,)).item())
            if k > 0:
                data_tensor = torch.rot90(data_tensor, k, [1, 2])
        # ---------------------------------------------------------
        
        target_modality_idx = self.modalities.index('T2')
        hr_image = data_tensor[target_modality_idx].unsqueeze(0) # [1, 256, 256]
        
        lr_hw = 64 if self.scale_factor == 4 else 128
        lr_images = []
        for i in range(len(self.modalities)):
            mod_hr = data_tensor[i].unsqueeze(0).unsqueeze(0) # [1, 1, 256, 256]
            mod_lr = torch.nn.functional.interpolate(
                mod_hr, 
                size=(lr_hw, lr_hw), 
                mode='bicubic', 
                align_corners=False
            )
            mod_lr = torch.clamp(mod_lr, 0.0, 1.0)
            lr_images.append(mod_lr.squeeze(0)) # [1, lr_hw, lr_hw]
            
        return lr_images, hr_image
