# src/augment/gan_augment.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random

# ----------------------------
# 数据集类（带标签）
# ----------------------------
class DefectDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}

        # 获取所有子目录作为类别
        class_dirs = [d for d in self.root_dir.iterdir() if d.is_dir()]
        class_dirs.sort()
        self.class_to_idx = {cls.name: idx for idx, cls in enumerate(class_dirs)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}

        for class_dir in class_dirs:
            label = self.class_to_idx[class_dir.name]
            for ext in ('*.png', '*.jpg', '*.jpeg'):
                for img_path in class_dir.rglob(ext):
                    self.image_paths.append(img_path)
                    self.labels.append(label)

        if len(self.image_paths) == 0:
            raise ValueError(f"❌ 目录 {root_dir} 中未找到任何图像！")
        print(f"✅ 加载 {len(self.image_paths)} 张图像，共 {len(self.class_to_idx)} 类：{list(self.class_to_idx.keys())}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            image = np.zeros((256, 256), dtype=np.uint8)
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# ----------------------------
# 条件生成器 (cGAN Generator)
# ----------------------------
class ConditionalGenerator(nn.Module):
    def __init__(self, nz=100, ngf=64, num_classes=6):
        super(ConditionalGenerator, self).__init__()
        self.nz = nz
        self.num_classes = num_classes
        # 将类别标签嵌入为 nz 维向量（与噪声同维）
        self.label_embedding = nn.Embedding(num_classes, nz)

        # 注意：输入通道数 = nz (noise) + nz (label) = nz * 2
        self.main = nn.Sequential(
            # 输入: [B, nz*2, 1, 1]
            nn.ConvTranspose2d(nz * 2, ngf * 32, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 32),
            nn.ReLU(True),
            # state size: (ngf*32) x 4 x 4
            nn.ConvTranspose2d(ngf * 32, ngf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            # state size: (ngf*16) x 8 x 8
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size: (ngf*8) x 16 x 16
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size: (ngf*4) x 32 x 32
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size: (ngf*2) x 64 x 64
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size: (ngf) x 128 x 128
            nn.ConvTranspose2d(ngf, 1, 4, 2, 1, bias=False),
            nn.Tanh()
            # output: [B, 1, 256, 256]
        )

    def forward(self, noise, labels):
        """
        noise: [B, nz, 1, 1]
        labels: [B]
        """
        # 嵌入标签 → [B, nz]
        label_emb = self.label_embedding(labels)
        # 展平噪声 → [B, nz]
        noise_flat = noise.squeeze(-1).squeeze(-1)  # 移除最后两个维度
        # 拼接 → [B, nz + nz] = [B, nz*2]
        x = torch.cat([noise_flat, label_emb], dim=1)
        # 重塑为 [B, nz*2, 1, 1]
        x = x.unsqueeze(-1).unsqueeze(-1)
        # 通过主干网络
        return self.main(x)

# ----------------------------
# 条件判别器 (cGAN Discriminator)
# ----------------------------
class ConditionalDiscriminator(nn.Module):
    def __init__(self, ndf=64, num_classes=6, img_size=256):
        super(ConditionalDiscriminator, self).__init__()
        self.num_classes = num_classes
        self.label_embedding = nn.Embedding(num_classes, img_size * img_size)

        self.main = nn.Sequential(
            nn.Conv2d(2, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 16, ndf * 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 32, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        label_emb = self.label_embedding(labels).view(img.size(0), 1, img.size(2), img.size(3))
        x = torch.cat([img, label_emb], dim=1)  # [B, 2, H, W]
        return self.main(x).view(-1, 1).squeeze(1)

# ----------------------------
# 权重初始化
# ----------------------------
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# ----------------------------
# 核心函数：训练 + 生成
# ----------------------------
def generate_gan_samples(
    image_dir,
    output_dir,
    num_samples=30,
    epochs=30,
    latent_dim=100,
    lr=0.0002
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 使用设备: {device}")

    # 数据预处理
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)  # [0,1] → [-1,1]
    ])

    dataset = DefectDataset(image_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    num_classes = len(dataset.class_to_idx)
    print(f"🏷️  共 {num_classes} 个类别: {dataset.idx_to_class}")

    # 初始化模型
    netG = ConditionalGenerator(nz=latent_dim, num_classes=num_classes).to(device)
    netD = ConditionalDiscriminator(num_classes=num_classes).to(device)
    netG.apply(weights_init)
    netD.apply(weights_init)

    # 损失与优化器
    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))

    # 训练循环
    print("⏳ 开始训练 cGAN...")
    for epoch in range(epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for real_images, labels in pbar:
            b_size = real_images.size(0)
            real_images = real_images.to(device)
            labels = labels.to(device)
            label_real = torch.full((b_size,), 1.0, dtype=torch.float, device=device)
            label_fake = torch.full((b_size,), 0.0, dtype=torch.float, device=device)

            # 更新判别器
            netD.zero_grad()
            output = netD(real_images, labels)
            errD_real = criterion(output, label_real)
            errD_real.backward()

            noise = torch.randn(b_size, latent_dim, 1, 1, device=device)
            fake = netG(noise, labels)
            output = netD(fake.detach(), labels)
            errD_fake = criterion(output, label_fake)
            errD_fake.backward()
            optimizerD.step()

            # 更新生成器
            netG.zero_grad()
            output = netD(fake, labels)
            errG = criterion(output, label_real)
            errG.backward()
            optimizerG.step()

    # 生成新图像（每类生成 num_samples // num_classes 张）
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    samples_per_class = max(1, num_samples // num_classes)
    print(f"🎨 每类生成 {samples_per_class} 张图像...")

    with torch.no_grad():
        for class_id in range(num_classes):
            class_name = dataset.idx_to_class[class_id]
            for i in range(samples_per_class):
                noise = torch.randn(1, latent_dim, 1, 1, device=device)
                label = torch.tensor([class_id], device=device)
                fake_img = netG(noise, label).cpu().squeeze().numpy()
                fake_img = (fake_img * 0.5 + 0.5) * 255
                fake_img = np.clip(fake_img, 0, 255).astype(np.uint8)
                out_path = Path(output_dir) / f"{class_name}_gan_{i:04d}.png"
                cv2.imwrite(str(out_path), fake_img)

    print(f"✅ cGAN 图像已保存至: {output_dir}")