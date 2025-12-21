import albumentations as A
import cv2
import os
from pathlib import Path


def get_traditional_pipeline(size=256):
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Rotate(limit=20, p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.GaussNoise(var_limit=(10.0, 30.0), p=0.3),
        A.Resize(size, size)
    ])


def apply_traditional_augmentation(image_dir, output_dir, num_samples=100, size=256):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    pipeline = get_traditional_pipeline(size)
    images = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg'))]

    count = 0
    while count < num_samples:
        img_path = np.random.choice(images)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        augmented = pipeline(image=img)["image"]
        out_path = os.path.join(output_dir, f"trad_{count:04d}.png")
        cv2.imwrite(out_path, augmented)
        count += 1
    print(f"✅ 传统增强完成，生成 {num_samples} 张图像")