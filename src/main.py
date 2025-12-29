# src/main.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.dataset_loader import load_and_preprocess_dataset
from src.augment.cgan_256 import train_cgan_256  # ← 导入新模块


def main():
    cfg = load_config()

    # 预处理（确保输出 256x256）
    processed_dir = load_and_preprocess_dataset(
        raw_dir=cfg['data']['raw_dir'],
        processed_dir=cfg['data']['processed_dir'],
        size=256,
        grayscale=True
    )

    if cfg['augmentation']['gan']['enable']:
        print("🔥 启动 256x256 cGAN 训练...")
        train_cgan_256(
            data_dir=processed_dir,
            output_dir=cfg['evaluation']['output_dir'] + "/cgan_256",
            epochs=cfg['augmentation']['gan'].get('epochs', 500),
            batch_size=cfg['augmentation']['gan'].get('batch_size', 4),
            nz=cfg['augmentation']['gan'].get('latent_dim', 100),
            lr=cfg['augmentation']['gan'].get('lr', 0.0001),
            save_interval=50,
            num_test_samples=1
        )

    print("\n🎉 完成！")


if __name__ == "__main__":
    main()