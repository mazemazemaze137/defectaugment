# src/dataset_loader.py
import cv2
import numpy as np
from pathlib import Path


def load_and_preprocess_dataset(raw_dir, processed_dir, size=256, grayscale=True):
    """
    加载并预处理数据集。

    参数:
        raw_dir (str): 原始数据目录，应包含子文件夹如 Crack/, Scratch/ 等
        processed_dir (str): 预处理后保存目录
        size (int): 目标图像尺寸（正方形）
        grayscale (bool): 是否转为灰度图

    返回:
        str: processed_dir 的绝对路径（关键！避免 main.py 传 None）
    """
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    if not raw_dir.exists():
        raise FileNotFoundError(f"❌ 原始数据目录不存在: {raw_dir}")

    # 支持的图像扩展名
    extensions = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff')

    total_images = 0

    # 遍历每个类别文件夹
    for class_dir in raw_dir.iterdir():
        if not class_dir.is_dir():
            continue

        out_class_dir = processed_dir / class_dir.name
        out_class_dir.mkdir(exist_ok=True)

        print(f"📁 处理类别: {class_dir.name}")
        count = 0

        for ext in extensions:
            for img_path in class_dir.glob(ext):
                try:
                    # 读取图像
                    if grayscale:
                        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                    else:
                        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)

                    if img is None:
                        print(f"⚠️ 跳过无效图像: {img_path}")
                        continue

                    # 调整尺寸
                    img_resized = cv2.resize(img, (size, size))

                    # 保存
                    out_path = out_class_dir / img_path.name
                    cv2.imwrite(str(out_path), img_resized)
                    count += 1
                    total_images += 1

                except Exception as e:
                    print(f"❌ 处理 {img_path} 时出错: {e}")
                    continue

        print(f"   → 已处理 {count} 张图像")

    print(f"\n✅ 数据预处理完成！共处理 {total_images} 张图像")
    print(f"📁 输出目录: {processed_dir.resolve()}")

    # ✅ 关键：返回字符串路径，避免 main.py 传 None
    return str(processed_dir)