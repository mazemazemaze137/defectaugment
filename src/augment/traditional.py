# src/augment/traditional.py
import albumentations as A
import cv2
import os
import numpy as np
import random
from pathlib import Path


def get_traditional_pipeline(size=256):
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Rotate(limit=20, p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.GaussNoise(noise_scale_factor=0.1, p=0.3),
        A.Resize(size, size)
    ])


def apply_traditional_augmentation(image_dir, output_dir, num_samples=100, size=256):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    pipeline = get_traditional_pipeline(size)

    # 递归收集所有图像路径
    image_paths = []
    for ext in ('*.png', '*.jpg', '*.jpeg'):
        image_paths.extend(Path(image_dir).rglob(ext))

    if not image_paths:
        raise ValueError(f"❌ No images found in {image_dir}")

    # 随机打乱顺序
    random.shuffle(image_paths)

    count = 0
    while count < num_samples:
        # 循环使用图像列表（避免索引越界）
        img_path = image_paths[count % len(image_paths)]

        # 读取图像
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # 应用增强
        augmented = pipeline(image=img)["image"]

        # === 关键：构造干净的文件名 ===
        # 获取相对于 image_dir 的路径
        try:
            rel_path = img_path.relative_to(image_dir)
            category = rel_path.parts[0]  # 如 "Crack"
            stem = rel_path.stem  # 如 "crack_0003"（不含 .png）
        except ValueError:
            # 如果无法计算相对路径（理论上不会发生），回退到简单命名
            category = "unknown"
            stem = img_path.stem

        out_name = f"{category}_{stem}_aug{count:04d}.png"
        out_path = output_path / out_name

        # 保存
        cv2.imwrite(str(out_path), augmented)
        print(f"[Debug] {out_name} ← {img_path}")  # 可清晰追踪来源
        count += 1

    print(f"✅ 传统增强完成，生成 {num_samples} 张图像")