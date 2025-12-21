import os
import cv2
import numpy as np
from pathlib import Path

def preprocess_image(img_path, size=256, grayscale=True):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR)
    img = cv2.resize(img, (size, size))
    # 可选：CLAHE 对比度增强
    if grayscale:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img = clahe.apply(img)
    return img

def load_and_preprocess_dataset(raw_dir, processed_dir, size=256, grayscale=True):
    Path(processed_dir).mkdir(parents=True, exist_ok=True)
    for img_name in os.listdir(raw_dir):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            src = os.path.join(raw_dir, img_name)
            dst = os.path.join(processed_dir, img_name)
            img = preprocess_image(src, size, grayscale)
            cv2.imwrite(dst, img)
    print(f"✅ 预处理完成，保存至 {processed_dir}")