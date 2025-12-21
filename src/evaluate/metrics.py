from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np
import os
import cv2


def calculate_ssim_psnr(original_dir, generated_dir):
    orig_files = sorted([f for f in os.listdir(original_dir) if f.endswith(('.png', '.jpg'))])
    gen_files = sorted([f for f in os.listdir(generated_dir) if f.endswith(('.png', '.jpg'))])

    ssim_vals, psnr_vals = [], []
    for o, g in zip(orig_files[:50], gen_files[:50]):  # 取前50对
        img_o = cv2.imread(os.path.join(original_dir, o), cv2.IMREAD_GRAYSCALE)
        img_g = cv2.imread(os.path.join(generated_dir, g), cv2.IMREAD_GRAYSCALE)
        ssim_vals.append(ssim(img_o, img_g))
        psnr_vals.append(psnr(img_o, img_g))

    return np.mean(ssim_vals), np.mean(psnr_vals)

# FID 需要单独用命令行工具：pytorch-fid
# 示例：pytorch_fid data/processed results/gan_generated --device cuda