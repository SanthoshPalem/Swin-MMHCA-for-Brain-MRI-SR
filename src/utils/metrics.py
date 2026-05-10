import torch
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure


def crop_border(sr, hr, scale=4):
    sr = torch.clamp(sr, 0.0, 1.0)
    hr = torch.clamp(hr, 0.0, 1.0)
    if scale > 0:
        sr = sr[:, :, scale:-scale, scale:-scale]
        hr = hr[:, :, scale:-scale, scale:-scale]
    return sr, hr


def calculate_psnr(sr, hr, scale=4, device=None, metric=None):
    sr, hr = crop_border(sr, hr, scale=scale)
    metric = metric or PeakSignalNoiseRatio(data_range=1.0).to(device or sr.device)
    return metric(sr, hr)


def calculate_ssim(sr, hr, scale=4, device=None, metric=None):
    sr, hr = crop_border(sr, hr, scale=scale)
    metric = metric or StructuralSimilarityIndexMeasure(data_range=1.0).to(device or sr.device)
    return metric(sr, hr)
