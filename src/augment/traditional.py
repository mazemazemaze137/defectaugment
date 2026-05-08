import random
from pathlib import Path

import albumentations as A
import cv2


def get_traditional_pipeline(size=256):
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Rotate(limit=20, p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.GaussNoise(noise_scale_factor=0.1, p=0.3),
            A.Resize(size, size),
        ]
    )


def apply_traditional_augmentation(image_dir, output_dir, num_samples=100, size=256, keep_class_dirs=True, seed=42):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    pipeline = get_traditional_pipeline(size)
    random.seed(seed)

    image_root = Path(image_dir)
    image_paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"):
        image_paths.extend(image_root.rglob(ext))

    if not image_paths:
        raise ValueError(f"No images found in {image_dir}")

    random.shuffle(image_paths)
    count = 0
    while count < num_samples:
        img_path = image_paths[count % len(image_paths)]
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            count += 1
            continue

        augmented = pipeline(image=img)["image"]
        try:
            rel_path = img_path.relative_to(image_root)
            category = rel_path.parts[0] if len(rel_path.parts) > 1 else "unknown"
            stem = rel_path.stem
        except ValueError:
            category = "unknown"
            stem = img_path.stem

        if keep_class_dirs:
            class_dir = output_path / category
            class_dir.mkdir(parents=True, exist_ok=True)
            out_path = class_dir / f"{stem}_traditional_{count:04d}.png"
        else:
            out_path = output_path / f"{category}_{stem}_traditional_{count:04d}.png"

        cv2.imwrite(str(out_path), augmented)
        count += 1

    print(f"Traditional augmentation finished. Generated {num_samples} images at {output_path}.")
    return str(output_path)
