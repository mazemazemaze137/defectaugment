# src/main.py
import sys
import os
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.dataset_loader import load_and_preprocess_dataset
from src.augment.cgan_256 import train_cgan_256


def main():
    cfg = load_config()
    gan_cfg = cfg['augmentation']['gan']
    image_size = gan_cfg.get('image_size', cfg['image'].get('size', 128))

    # 1. 预处理
    print("🔄 开始数据预处理...")
    processed_dir = load_and_preprocess_dataset(
        raw_dir=cfg['data']['raw_dir'],
        processed_dir=cfg['data']['processed_dir'],
        size=image_size,
        grayscale=cfg['image'].get('grayscale', True)
    )

    # 2. GAN 训练
    if gan_cfg['enable']:
        print(f"\n🔥 启动 {image_size}x{image_size} cGAN 训练...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_output_dir = os.path.join(cfg['evaluation']['output_dir'], f"gan_run_{timestamp}")
        print(f"📁 输出目录: {run_output_dir}")

        train_cgan_256(
            data_dir=processed_dir,
            output_dir=run_output_dir,
            epochs=gan_cfg.get('epochs', 500),
            batch_size=gan_cfg.get('batch_size', 4),
            nz=gan_cfg.get('latent_dim', 100),
            lr=gan_cfg.get('lr', 0.0002),
            lr_g=gan_cfg.get('lr_g'),
            lr_d=gan_cfg.get('lr_d'),
            image_size=image_size,
            save_interval=gan_cfg.get('save_interval', 20),
            num_test_samples=gan_cfg.get('num_test_samples', 8),
            resume=False
        )

    print("\n🎉 所有任务完成！")


if __name__ == "__main__":
    main()
