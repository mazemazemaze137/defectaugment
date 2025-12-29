# src/augment/simple_gan.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


class SimpleGenerator(nn.Module):
    def __init__(self, nz=100):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, 64, 4, 1, 0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)


class SimpleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 8, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(8, 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x).view(-1, 1).squeeze(1)


def train_simple_gan(image_dir, output_dir, epochs=10, batch_size=4, nz=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 使用设备: {device}")

    # === 1. 加载所有图像 ===
    transform = lambda x: (x.astype(np.float32) / 255.0 - 0.5) * 2.0  # [0,255] → [-1,1]
    images = []
    extensions = ('*.png', '*.jpg', '*.jpeg', '*.bmp')

    for ext in extensions:
        for img_path in Path(image_dir).rglob(ext):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (64, 64))
                img = transform(img)
                images.append(img)

    if len(images) == 0:
        raise ValueError(f"❌ 在 {image_dir} 中未找到任何图像！")

    # 转为 (N, 1, H, W)
    images = np.stack(images, axis=0)  # (N, 64, 64)
    images = np.expand_dims(images, axis=1)  # (N, 1, 64, 64)
    tensor_images = torch.from_numpy(images).float()

    # === 2. 创建 DataLoader ===
    dataset = TensorDataset(tensor_images)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    print(f"✅ 加载 {len(images)} 张图像，形状: {images.shape}")

    # === 3. 初始化模型 ===
    netG = SimpleGenerator(nz).to(device)
    netD = SimpleDiscriminator().to(device)
    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # === 4. 训练循环 ===
    for epoch in range(epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
        for (real_images,) in pbar:  # ✅ 解包 tuple
            real_images = real_images.to(device)
            b_size = real_images.size(0)

            # 真实标签
            label_real = torch.ones(b_size, device=device)
            label_fake = torch.zeros(b_size, device=device)

            # --- 更新判别器 ---
            netD.zero_grad()
            output = netD(real_images)
            errD_real = criterion(output, label_real)
            errD_real.backward()

            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake = netG(noise)
            output = netD(fake.detach())
            errD_fake = criterion(output, label_fake)
            errD_fake.backward()
            optimizerD.step()

            # --- 更新生成器 ---
            netG.zero_grad()
            output = netD(fake)
            errG = criterion(output, label_real)
            errG.backward()
            optimizerG.step()

    # === 5. 保存生成图像 ===
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        noise = torch.randn(1, nz, 1, 1, device=device)
        fake_img = netG(noise).cpu().squeeze().numpy()
        fake_img = (fake_img * 0.5 + 0.5) * 255  # [-1,1] → [0,255]
        fake_img = np.clip(fake_img, 0, 255).astype(np.uint8)
        cv2.imwrite(f"{output_dir}/simple_gan_test.png", fake_img)

    print(f"✅ 简化 GAN 测试图像已保存至: {output_dir}")