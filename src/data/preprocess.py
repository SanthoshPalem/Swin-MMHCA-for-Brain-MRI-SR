import os
import numpy as np
import torch
import torch.nn.functional as F
import SimpleITK as sitk
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import random

class MRIPreProcessor:
    """
    RESEARCH-GRADE MRI PIPELINE (SimpleITK Native)
    ✔ Metadata preservation (Origin, Direction, Spacing)
    ✔ Multi-modal Registration (T1/PD -> T2)
    ✔ DGX Optimized (Single-threaded, Sampling)
    ✔ Resizing, Slicing, Normalization
    """

    def __init__(self, threshold=0.01, target_shape=(256, 256, 128),
                 pixel_threshold=0.10, coverage_threshold=0.28):
        self.threshold = threshold
        self.target_shape = target_shape
        # Brain coverage filter: reject slices where bright pixels (brain)
        # cover less than `coverage_threshold` fraction of the image.
        # This removes inferior slices (cerebellum/brainstem) that have a
        # small brain cross-section surrounded by a large black background.
        self.pixel_threshold = pixel_threshold
        self.coverage_threshold = coverage_threshold

    # -----------------------------
    # STEP 1: Native SITK Loading
    # -----------------------------
    def load_sitk(self, path):
        """Loads image and forces RAS orientation while preserving metadata."""
        img = sitk.ReadImage(path)
        img = sitk.DICOMOrient(img, "RAS")
        # 🔥 CRITICAL FIX: Cast to Float32 (Registration filters don't support 16-bit int)
        img = sitk.Cast(img, sitk.sitkFloat32)
        return img

    # -----------------------------
    # STEP 2: Native SITK Registration (ROBUST VERSION)
    # -----------------------------
    def register(self, fixed, moving):
        """
        Robust SimpleITK Registration.
        Uses Multi-resolution strategy with Mattes Mutual Information.
        """
        # DGX SPEED FIX: Limit threads per subject
        sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)

        # 1. Initial Transform (GEOMETRY-based is more stable for brain scans)
        initial_transform = sitk.CenteredTransformInitializer(
            fixed, moving,
            sitk.AffineTransform(3),
            sitk.CenteredTransformInitializerFilter.GEOMETRY
        )

        # 2. Registration Method Setup
        registration = sitk.ImageRegistrationMethod()

        # Metric: Mattes Mutual Information (Multi-modal standard)
        registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        
        # Increase Sampling to 15% for better accuracy
        registration.SetMetricSamplingPercentage(0.15, seed=42)
        registration.SetMetricSamplingStrategy(registration.RANDOM)

        # Multi-resolution (Coarse -> Fine)
        registration.SetShrinkFactorsPerLevel([4, 2, 1])
        registration.SetSmoothingSigmasPerLevel([2, 1, 0])
        registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

        # Optimizer: Regular Step Gradient Descent
        # Add OptimizerScalesFromPhysicalShift for stability
        registration.SetOptimizerAsRegularStepGradientDescent(
            learningRate=1.0,  # Lower learning rate for more precise convergence
            minStep=1e-5,
            numberOfIterations=250 # More iterations
        )
        registration.SetOptimizerScalesFromPhysicalShift()

        registration.SetInterpolator(sitk.sitkLinear)
        registration.SetInitialTransform(initial_transform, inPlace=False)

        # 3. Execute Registration
        try:
            final_transform = registration.Execute(fixed, moving)
        except Exception as e:
            print(f"      [!] Registration failed, falling back to identity: {e}")
            return moving

        # 4. Resample moving image into fixed image grid
        resampled = sitk.Resample(
            moving, fixed, final_transform,
            sitk.sitkLinear, 0.0, moving.GetPixelID()
        )

        return resampled

    # -----------------------------
    # STEP 3: Resize (Volume)
    # -----------------------------
    def resize_volume(self, data):
        """Standardizes volume to target grid shape."""
        tensor = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)
        resized = F.interpolate(
            tensor,
            size=self.target_shape,
            mode='trilinear',
            align_corners=False
        )
        return resized.squeeze().numpy()

    # -----------------------------
    # STEP 4: Normalize (2D)
    # -----------------------------
    def normalize(self, img):
        mn, mx = img.min(), img.max()
        return (img - mn) / (mx - mn) if mx > mn else np.zeros_like(img)

    # -----------------------------
    # STEP 5: Process Subject
    # -----------------------------
    def process_subject(self, t1_path, t2_path, pd_path):
        """Full Native SITK Pipeline: Load -> Register -> Array -> Resize -> Stack."""
        
        # 1. Load (Native SITK objects)
        t1 = self.load_sitk(t1_path)
        t2 = self.load_sitk(t2_path)
        pd = self.load_sitk(pd_path)

        # 2. Register (T2 is reference)
        print("      - Registering T1 to T2...")
        t1_reg = self.register(t2, t1)
        print("      - Registering PD to T2...")
        pd_reg = self.register(t2, pd)

        # 3. Convert to Numpy AFTER Registration (to keep metadata during alignment)
        # SITK gives (D, H, W) -> Transpose to (H, W, D) for resizing logic
        v_t1 = np.transpose(sitk.GetArrayFromImage(t1_reg), (1, 2, 0))
        v_t2 = np.transpose(sitk.GetArrayFromImage(t2), (1, 2, 0))
        v_pd = np.transpose(sitk.GetArrayFromImage(pd_reg), (1, 2, 0))

        # 4. Resize all to standardized grid
        v_t1 = self.resize_volume(v_t1.astype(np.float32))
        v_t2 = self.resize_volume(v_t2.astype(np.float32))
        v_pd = self.resize_volume(v_pd.astype(np.float32))

        # 5. Slice and Multi-modal Stack
        depth = self.target_shape[2]
        valid_stacks = []

        for z in range(depth):
            s_t1, s_t2, s_pd = v_t1[:, :, z], v_t2[:, :, z], v_pd[:, :, z]

            # Normalize T2 slice for consistent threshold comparisons
            s_t2_norm = self.normalize(s_t2)

            # Filter 1: reject near-empty/dark slices (top & bottom skull caps)
            if np.mean(s_t2_norm) < self.threshold:
                continue

            # Filter 2: reject inferior brain slices (cerebellum / brainstem)
            # These slices have a small brain cross-section in a large black field.
            # Good cortical slices fill >28% of the image with bright brain tissue.
            total_pixels = s_t2_norm.size
            brain_pixels = np.sum(s_t2_norm > self.pixel_threshold)
            coverage = brain_pixels / total_pixels
            if coverage < self.coverage_threshold:
                continue

            valid_stacks.append(np.stack([
                self.normalize(s_t1),
                s_t2_norm,
                self.normalize(s_pd)
            ], axis=0))

        return valid_stacks

# -----------------------------
# DATASET CLASS
# -----------------------------
class MultiModalMRIDataset(Dataset):
    def __init__(self, data_dir, augment=True):
        self.data_dir = data_dir
        self.files = sorted([f for f in os.listdir(data_dir) if f.endswith(".pt")])
        self.augment_flag = augment

    def __len__(self):
        return len(self.files)

    def augment(self, x):
        if random.random() > 0.5: x = TF.hflip(x)
        if random.random() > 0.5: x = TF.vflip(x)
        angle = random.choice([0, 90, 180, 270])
        if angle != 0: x = TF.rotate(x, angle)
        return x

    def __getitem__(self, idx):
        path = os.path.join(self.data_dir, self.files[idx])
        tensor = torch.load(path, weights_only=True).float()
        if self.augment_flag: tensor = self.augment(tensor)
        return tensor

if __name__ == "__main__":
    print("🔥 NATIVE SIMPLE-ITK PIPELINE READY 🔥")
