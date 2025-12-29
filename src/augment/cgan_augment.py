# src/augment/cgan_augment.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random

# ----------------------------
# 条件判别器
# ----------------------------
class ConditionalDiscriminator(nn.Module):
    def __init__(self, ndf=64, num_classes=6):
        super().__init__()
        self.num_classes = num_classes
        self.label_embedding = nn.Embedding(num_classes, 128 * 128)
        self.main = nn.Sequential(
            nn.Conv2d(2, ndf, 4, 2, 1, bias=False),         # 64
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),   # 32
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False), # 16
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False), # 8
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 8, 1, 0, bias=False),      # 1x1
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        label_emb = self.label_embedding(labels).view(-1, 1, 128, 128)
        x = torch.cat([img, label_emb], dim=1)
        return self.main(x).view(-1)

# ----------------------------
# 条件生成器
# ----------------------------
class ConditionalGenerator(nn.Module):
    def __init__(self, nz=100, ngf=64, num_classes=6):
        super().__init__()
        self.nz = nz
        self.num_classes = num_classes
        self.label_embedding = nn.Embedding(num_classes, nz)
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz * 2, ngf * 8, 4, 1, 0, bias=False),  # 4x4
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),  # 8x8
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),  # 16x16
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),      # 32x32
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, ngf // 2, 4, 2, 1, bias=False),     # 64x64
            nn.BatchNorm2d(ngf // 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf // 2, 1, 4, 2, 1, bias=False),       # 128x128
            nn.Tanh()
        )

    def forward(self, noise, labels):
        label_emb = self.label_embedding(labels).unsqueeze(-1).unsqueeze(-1)
        x = torch.cat([noise, label_emb], dim=1)
        return self.main(x)

# ----------------------------
# 自定义数据集（带类别标签）
# ----------------------------
class DefectDataset(Dataset):
    def __init__(self, root_dir, size=64):
        self.root_dir = Path(root_dir)
        self.size = size
        self.class_names = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}
        self.img_paths = []
        self.labels = []

        for class_name in self.class_names:
            class_dir = self.root_dir / class_name
            for ext in ('*.png', '*.jpg', '*.jpeg'):
                for p in class_dir.glob(ext):
                    self.img_paths.append(p)
                    self.labels.append(self.class_to_idx[class_name])

        print(f"✅ 加载 {len(self.img_paths)} 张图像，共 {len(self.class_names)} 类")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = cv2.imread(str(self.img_paths[idx]), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (self.size, self.size))
        img = (img.astype(np.float32) / 255.0 - 0.5) * 2.0  # [-1, 1]
        img = np.expand_dims(img, axis=0)  # [1, H, W]
        label = self.labels[idx]
        return torch.from_numpy(img), torch.tensor(label, dtype=torch.long)

# ----------------------------
# 主训练函数
# ----------------------------
def train_cgan(
    data_dir,
    output_dir,
    epochs=200,
    batch_size=16,
    nz=100,
    lr=0.0002,
    save_interval=20,
    num_test_samples=1  # 每类只生成 1 张用于测试
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 使用设备: {device}")

    # 数据集
    dataset = DefectDataset(data_dir, size=64)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    num_classes = len(dataset.class_names)

    # 模型
    netG = ConditionalGenerator(nz=nz, num_classes=num_classes).to(device)
    netD = ConditionalDiscriminator(num_classes=num_classes).to(device)

    # 优化器
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    # 固定噪声用于可视化（每类 1 个）
    fixed_noise = torch.randn(num_classes * num_test_samples, nz, 1, 1, device=device)
    fixed_labels = torch.arange(num_classes).repeat_interleave(num_test_samples).to(device)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 训练
    for epoch in range(1, epochs + 1):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")
        for real_imgs, labels in pbar:
            real_imgs = real_imgs.to(device)
            labels = labels.to(device)
            b_size = real_imgs.size(0)

            # --- 判别器更新 ---
            netD.zero_grad()
            # 真实样本（使用软标签：0.9 而不是 1.0）
            label_real = torch.full((b_size,), 0.9, device=device)
            output = netD(real_imgs, labels)
            errD_real = criterion(output, label_real)
            errD_real.backward()

            # 生成样本
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake = netG(noise, labels)
            label_fake = torch.zeros(b_size, device=device)
            output = netD(fake.detach(), labels)
            errD_fake = criterion(output, label_fake)
            errD_fake.backward()
            optimizerD.step()

            # --- 生成器更新 ---
            netG.zero_grad()
            label_g = torch.ones(b_size, device=device)  # 欺骗判别器
            output = netD(fake, labels)
            errG = criterion(output, label_g)
            errG.backward()
            optimizerG.step()

            pbar.set_postfix(D_real=errD_real.item(), D_fake=errD_fake.item(), G=errG.item())

        # 定期保存生成样本
        if epoch % save_interval == 0 or epoch == epochs:
            with torch.no_grad():
                fake_imgs = netG(fixed_noise, fixed_labels).cpu()
                grid = []
                for i in range(num_classes):
                    img = fake_imgs[i].squeeze().numpy()
                    img = (img * 0.5 + 0.5) * 255
                    img = np.clip(img, 0, 255).astype(np.uint8)
                    cv2.imwrite(f"{output_dir}/epoch_{epoch:03d}_class_{i}.png", img)
            print(f"🖼️  已保存第 {epoch} 轮生成样本")

    print(f"✅ cGAN 训练完成！结果保存至: {output_dir}")