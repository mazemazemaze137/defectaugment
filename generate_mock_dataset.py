# generate_mock_dataset.py
import os
import numpy as np
import cv2
from pathlib import Path
import random

# 缺陷类别
CLASSES = [
    "Crack",
    "Inclusion",
    "Patches",
    "Pitted_Surface",
    "Rolled_In_Scale",
    "Scratch"
]

IMG_SIZE = 200
NUM_PER_CLASS = 50
OUTPUT_DIR = "data/raw/mock_defects"


def add_noise(img, intensity=10):
    noise = np.random.normal(0, intensity, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def generate_crack():
    img = np.ones((IMG_SIZE, IMG_SIZE), dtype=np.uint8) * 200  # 浅灰背景
    for _ in range(random.randint(1, 3)):
        x1, y1 = random.randint(0, IMG_SIZE), random.randint(0, IMG_SIZE)
        x2, y2 = random.randint(0, IMG_SIZE), random.randint(0, IMG_SIZE)
        thickness = random.randint(1, 3)
        cv2.line(img, (x1, y1), (x2, y2), color=50, thickness=thickness)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    return add_noise(img)


def generate_inclusion():
    img = np.ones((IMG_SIZE, IMG_SIZE), dtype=np.uint8) * 200
    for _ in range(random.randint(1, 4)):
        cx, cy = random.randint(20, IMG_SIZE - 20), random.randint(20, IMG_SIZE - 20)
        axes = (random.randint(5, 15), random.randint(5, 15))
        angle = random.randint(0, 180)
        cv2.ellipse(img, (cx, cy), axes, angle, 0, 360, color=80, thickness=-1)
    return add_noise(img)


def generate_patches():
    img = np.ones((IMG_SIZE, IMG_SIZE), dtype=np.uint8) * 200
    for _ in range(random.randint(2, 5)):
        pts = np.array([[random.randint(0, IMG_SIZE) for _ in range(2)] for _ in range(5)], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(img, [pts], color=70)
    return add_noise(img)


def generate_pitted_surface():
    img = np.ones((IMG_SIZE, IMG_SIZE), dtype=np.uint8) * 200
    for _ in range(random.randint(20, 50)):
        x, y = random.randint(5, IMG_SIZE - 5), random.randint(5, IMG_SIZE - 5)
        r = random.randint(2, 5)
        cv2.circle(img, (x, y), r, color=150, thickness=-1)
    kernel = np.ones((3, 3), np.float32) / 9
    img = cv2.filter2D(img, -1, kernel)
    return add_noise(img)


def generate_rolled_in_scale():
    img = np.ones((IMG_SIZE, IMG_SIZE), dtype=np.uint8) * 200
    num_bands = random.randint(3, 8)
    for i in range(num_bands):
        start_x = random.randint(0, IMG_SIZE // 2)
        end_x = start_x + random.randint(30, 80)
        if end_x > IMG_SIZE: end_x = IMG_SIZE
        band_height = random.randint(8, 20)
        y = random.randint(band_height, IMG_SIZE - band_height)
        cv2.rectangle(img, (start_x, y - band_height // 2), (end_x, y + band_height // 2), color=100, thickness=-1)
    return add_noise(img)


def generate_scratch():
    img = np.ones((IMG_SIZE, IMG_SIZE), dtype=np.uint8) * 200
    for _ in range(random.randint(1, 3)):
        points = []
        num_pts = random.randint(5, 15)
        for i in range(num_pts):
            x = int(IMG_SIZE * i / (num_pts - 1))
            y = random.randint(50, 150)
            points.append([x, y + random.randint(-20, 20)])
        points = np.array(points, np.int32)
        for i in range(len(points) - 1):
            cv2.line(img, tuple(points[i]), tuple(points[i + 1]), color=60, thickness=random.randint(1, 2))
    return add_noise(img)


GENERATORS = {
    "Crack": generate_crack,
    "Inclusion": generate_inclusion,
    "Patches": generate_patches,
    "Pitted_Surface": generate_pitted_surface,
    "Rolled_In_Scale": generate_rolled_in_scale,
    "Scratch": generate_scratch
}


def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    for cls_name in CLASSES:
        class_dir = os.path.join(OUTPUT_DIR, cls_name)
        Path(class_dir).mkdir(exist_ok=True)
        generator = GENERATORS[cls_name]
        print(f"Generating {cls_name} ({NUM_PER_CLASS} images)...")
        for i in range(NUM_PER_CLASS):
            img = generator()
            cv2.imwrite(os.path.join(class_dir, f"{cls_name.lower()}_{i:04d}.png"), img)

    print(f"✅ Mock dataset generated at: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()