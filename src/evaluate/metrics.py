from pathlib import Path

import cv2
import numpy as np
import torch
from pytorch_fid.fid_score import (
    calculate_activation_statistics,
    calculate_frechet_distance,
)
from pytorch_fid.inception import InceptionV3
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def collect_image_paths(image_dir):
    image_dir = Path(image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    return sorted(
        path
        for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _read_grayscale(path, image_size=256):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return cv2.resize(img, (image_size, image_size))


def calculate_ssim_psnr(real_dir, generated_dir, max_pairs=200, image_size=256):
    real_paths = collect_image_paths(real_dir)
    generated_paths = collect_image_paths(generated_dir)
    pair_count = min(len(real_paths), len(generated_paths), max_pairs)
    if pair_count == 0:
        raise ValueError("No comparable image pairs found.")

    ssim_vals = []
    psnr_vals = []
    for real_path, generated_path in zip(real_paths[:pair_count], generated_paths[:pair_count]):
        real_img = _read_grayscale(real_path, image_size=image_size)
        generated_img = _read_grayscale(generated_path, image_size=image_size)
        if real_img is None or generated_img is None:
            continue
        ssim_vals.append(ssim(real_img, generated_img, data_range=255))
        psnr_vals.append(psnr(real_img, generated_img, data_range=255))

    if not ssim_vals:
        raise ValueError("Images were found, but none could be decoded for SSIM/PSNR.")

    return {
        "ssim": float(np.mean(ssim_vals)),
        "psnr": float(np.mean(psnr_vals)),
        "pairs": len(ssim_vals),
    }


def calculate_fid(real_dir, generated_dir, batch_size=32, dims=2048, num_workers=0, device=None):
    real_paths = collect_image_paths(real_dir)
    generated_paths = collect_image_paths(generated_dir)
    if not real_paths:
        raise ValueError(f"No real images found under: {real_dir}")
    if not generated_paths:
        raise ValueError(f"No generated images found under: {generated_dir}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)

    mu_real, sigma_real = calculate_activation_statistics(
        real_paths,
        model,
        batch_size=batch_size,
        dims=dims,
        device=device,
        num_workers=num_workers,
    )
    mu_generated, sigma_generated = calculate_activation_statistics(
        generated_paths,
        model,
        batch_size=batch_size,
        dims=dims,
        device=device,
        num_workers=num_workers,
    )
    return float(calculate_frechet_distance(mu_real, sigma_real, mu_generated, sigma_generated))


def evaluate_generated_dataset(
    real_dir,
    generated_dir,
    max_pairs=200,
    image_size=256,
    fid_batch_size=32,
    fid_dims=2048,
    fid_num_workers=0,
    calculate_fid_metric=True,
):
    real_count = len(collect_image_paths(real_dir))
    generated_count = len(collect_image_paths(generated_dir))
    pair_metrics = calculate_ssim_psnr(
        real_dir,
        generated_dir,
        max_pairs=max_pairs,
        image_size=image_size,
    )

    result = {
        "real_count": real_count,
        "generated_count": generated_count,
        **pair_metrics,
    }
    if calculate_fid_metric:
        result["fid"] = calculate_fid(
            real_dir,
            generated_dir,
            batch_size=fid_batch_size,
            dims=fid_dims,
            num_workers=fid_num_workers,
        )
    return result
