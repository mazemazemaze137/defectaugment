from src.config import load_config
from src.dataset_loader import load_and_preprocess_dataset
from src.augment.traditional import apply_traditional_augmentation
from src.augment.gan_augment import train_gan, generate_with_gan
import numpy as np
import cv2
import os


def main():
    cfg = load_config()

    # 1. 数据预处理
    load_and_preprocess_dataset(
        cfg['data']['raw_dir'],
        cfg['data']['processed_dir'],
        size=cfg['image']['size'],
        grayscale=cfg['image']['grayscale']
    )

    # 2. 传统增强
    if cfg['augmentation']['traditional']['enable']:
        apply_traditional_augmentation(
            cfg['data']['processed_dir'],
            "results/traditional",
            num_samples=cfg['augmentation']['traditional']['num_samples'],
            size=cfg['image']['size']
        )

    # 3. GAN 增强（简化：加载部分真实图像用于训练）
    if cfg['augmentation']['gan']['enable']:
        # 加载少量真实图像用于 GAN 训练（实际应使用完整数据集）
        real_imgs = []
        for f in os.listdir(cfg['data']['processed_dir'])[:200]:
            img = cv2.imread(os.path.join(cfg['data']['processed_dir'], f), cv2.IMREAD_GRAYSCALE)
            real_imgs.append(img)
        real_imgs = np.array(real_imgs)

        generator = train_gan(
            real_imgs,
            latent_dim=cfg['augmentation']['gan']['latent_dim'],
            epochs=cfg['augmentation']['gan']['epochs'],
            lr=cfg['augmentation']['gan']['lr']
        )
        generate_with_gan(
            generator,
            num_samples=cfg['augmentation']['gan']['num_samples'],
            latent_dim=cfg['augmentation']['gan']['latent_dim'],
            output_dir="results/gan_generated"
        )

    print("🎉 系统流程执行完毕！请查看 results/ 目录。")


if __name__ == "__main__":
    main()