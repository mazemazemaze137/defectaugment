# src/augment/cgan_256.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import os

# ----------------------------
# 条件判别器 (256x256)
# ----------------------------
class ConditionalDiscriminator(nn.Module):
    def __init__(self, ndf=32, num_classes=6):
        super().__init__()
        self.num_classes = num_classes
        # 嵌入向量将被 reshape 为 [B, 1, 256, 256]
        self.label_embedding = nn.Embedding(num_classes, 256 * 256)
        self.main = nn.Sequential(
            # 输入: [B, 2, 256, 256]
            nn.Conv2d(2, ndf, 4, 2, 1, bias=False),          # 128
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),    # 64
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False), # 32
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False), # 16
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False), # 8
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 16, 1, 8, 1, 0, bias=False),     # 1x1
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        # img: [B, 1, 256, 256]
        # labels: [B]
        label_emb = self.label_embedding(labels).view(-1, 1, 256, 256)
        x = torch.cat([img, label_emb], dim=1)  # [B, 2, 256, 256]
        return self.main(x).view(-1)


# ----------------------------
# 条件生成器 (256x256)
# ----------------------------
class ConditionalGenerator(nn.Module):
    def __init__(self, nz=100, ngf=32, num_classes=6):
        super().__init__()
        self.nz = nz
        self.num_classes = num_classes
        self.label_embedding = nn.Embedding(num_classes, nz)
        self.main = nn.Sequential(
            # [B, nz*2, 1, 1] → [B, ngf*16, 4, 4]
            nn.ConvTranspose2d(nz * 2, ngf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),  # 8x8
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),   # 16x16
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),   # 32x32
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),       # 64x64
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, ngf // 2, 4, 2, 1, bias=False),      # 128x128
            nn.BatchNorm2d(ngf // 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf // 2, 1, 4, 2, 1, bias=False),        # 256x256
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # noise: [B, nz, 1, 1]
        # labels: [B]
        label_emb = self.label_embedding(labels).unsqueeze(-1).unsqueeze(-1)  # [B, nz, 1, 1]
        x = torch.cat([noise, label_emb], dim=1)  # [B, nz*2, 1, 1]
        return self.main(x)


# ----------------------------
# 数据集（支持任意子文件夹）
# ----------------------------
class DefectDataset(Dataset):
    def __init__(self, root_dir, size=256):
        self.root_dir = Path(root_dir)
        self.size = size
        if not self.root_dir.exists():
            raise FileNotFoundError(f"❌ 目录不存在: {root_dir}")

        self.class_names = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        if not self.class_names:
            raise ValueError(f"❌ {root_dir} 中没有类别子文件夹！")

        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}
        self.img_paths = []
        self.labels = []

        for class_name in self.class_names:
            class_dir = self.root_dir / class_name
            for ext in ('*.png', '*.jpg', '*.jpeg', '*.bmp'):
                for p in class_dir.glob(ext):
                    self.img_paths.append(p)
                    self.labels.append(self.class_to_idx[class_name])

        if len(self.img_paths) == 0:
            raise ValueError(f"❌ 在 {root_dir} 中未找到任何图像！")

        print(f"✅ 加载 {len(self.img_paths)} 张图像，共 {len(self.class_names)} 类")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = cv2.imread(str(self.img_paths[idx]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            # 跳过损坏图像
            return self.__getitem__((idx + 1) % len(self))
        img = cv2.resize(img, (self.size, self.size))
        img = (img.astype(np.float32) / 255.0 - 0.5) * 2.0  # [-1, 1]
        img = np.expand_dims(img, axis=0)  # [1, H, W]
        label = self.labels[idx]
        return torch.from_numpy(img), torch.tensor(label, dtype=torch.long)


# ----------------------------
# 训练函数
# ----------------------------
def train_cgan_256(
    data_dir,
    output_dir,
    epochs=500,
    batch_size=4,      # 256x256 下建议 2~4
    nz=100,
    lr=0.0001,
    save_interval=50,
    num_test_samples=1
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 使用设备: {device}")
    print(f"⚠️  注意：256x256 GAN 需要大量显存，请确保 batch_size 合理")

    # 数据集
    dataset = DefectDataset(data_dir, size=256)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    num_classes = len(dataset.class_names)

    # 模型
    netG = ConditionalGenerator(nz=nz, ngf=32, num_classes=num_classes).to(device)
    netD = ConditionalDiscriminator(ndf=32, num_classes=num_classes).to(device)

    # 初始化权重
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    netG.apply(weights_init)
    netD.apply(weights_init)

    # 优化器
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    # 固定噪声用于可视化
    fixed_noise = torch.randn(num_classes * num_test_samples, nz, 1, 1, device=device)
    fixed_labels = torch.arange(num_classes).repeat_interleave(num_test_samples).to(device)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 训练循环
    for epoch in range(1, epochs + 1):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")
        for real_imgs, labels in pbar:
            real_imgs = real_imgs.to(device)
            labels = labels.to(device)
            b_size = real_imgs.size(0)

            # --- 更新判别器 ---
            netD.zero_grad()
            label_real = torch.full((b_size,), 0.9, device=device)  # 标签平滑
            output = netD(real_imgs, labels)
            errD_real = criterion(output, label_real)
            errD_real.backward()

            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake = netG(noise, labels)
            label_fake = torch.zeros(b_size, device=device)
            output = netD(fake.detach(), labels)
            errD_fake = criterion(output, label_fake)
            errD_fake.backward()
            optimizerD.step()

            # --- 更新生成器 ---
            netG.zero_grad()
            label_g = torch.ones(b_size, device=device)
            output = netD(fake, labels)
            errG = criterion(output, label_g)
            errG.backward()
            optimizerG.step()

            pbar.set_postfix(D_real=errD_real.item(), D_fake=errD_fake.item(), G=errG.item())

        # 保存生成样本
        if epoch % save_interval == 0 or epoch == epochs:
            with torch.no_grad():
                fake_imgs = netG(fixed_noise, fixed_labels).cpu()
                for i in range(num_classes):
                    img = fake_imgs[i].squeeze().numpy()
                    img = (img * 0.5 + 0.5) * 255
                    img = np.clip(img, 0, 255).astype(np.uint8)
                    out_path = Path(output_dir) / f"epoch_{epoch:04d}_class_{dataset.class_names[i]}.png"
                    cv2.imwrite(str(out_path), img)
            print(f"🖼️  已保存第 {epoch} 轮生成样本")

    # 保存最终模型（可选）
    torch.save(netG.state_dict(), os.path.join(output_dir, "generator_final.pth"))
    torch.save(netD.state_dict(), os.path.join(output_dir, "discriminator_final.pth"))
    print(f"✅ 训练完成！结果保存至: {output_dir}")