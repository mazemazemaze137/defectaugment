import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
from pathlib import Path


# --- Generator ---
class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_size=256):
        super().__init__()
        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


# --- 简化训练函数（实际使用时需加载真实数据）---
def train_gan(real_images, latent_dim=100, epochs=10, lr=0.0002, device='cpu'):
    generator = Generator(latent_dim).to(device)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))

    real_tensor = torch.from_numpy(real_images).float().unsqueeze(1).to(device) / 127.5 - 1  # [-1,1]
    dataset = TensorDataset(real_tensor)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    for epoch in range(epochs):
        for i, (imgs,) in enumerate(dataloader):
            z = torch.randn(imgs.size(0), latent_dim).to(device)
            gen_imgs = generator(z)
            # 这里省略判别器和对抗损失（简化版仅用重建损失示意）
            loss = torch.mean((gen_imgs - imgs) ** 2)  # 仅作示意，实际应使用 GAN loss
            optimizer_G.zero_grad()
            loss.backward()
            optimizer_G.step()
        print(f"Epoch [{epoch + 1}/{epochs}]")

    return generator


def generate_with_gan(generator, num_samples=100, latent_dim=100, output_dir="results/gan_generated", device='cpu'):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    generator.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim).to(device)
        gen_imgs = generator(z).cpu().numpy()
        gen_imgs = ((gen_imgs + 1) * 127.5).astype(np.uint8)
        for i in range(num_samples):
            cv2.imwrite(os.path.join(output_dir, f"gan_{i:04d}.png"), gen_imgs[i, 0])
    print(f"✅ GAN 生成完成，保存至 {output_dir}")