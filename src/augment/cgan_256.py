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
import csv  # ✅ 新增：用于记录训练日志


# ----------------------------
# 1. 改进后的条件判别器 (带 Spectral Normalization)
# ----------------------------
class ConditionalDiscriminator(nn.Module):
    def __init__(self, ndf=32, num_classes=6):
        super().__init__()
        self.num_classes = num_classes
        self.label_embedding = nn.Embedding(num_classes, 256 * 256)

        def sn_conv2d(in_c, out_c, k, s, p):
            """应用谱归一化的卷积层，极大提升训练稳定性"""
            return nn.utils.spectral_norm(nn.Conv2d(in_c, out_c, k, s, p, bias=False))

        self.main = nn.Sequential(
            sn_conv2d(2, ndf, 4, 2, 1),  # 128
            nn.LeakyReLU(0.2, inplace=True),

            sn_conv2d(ndf, ndf * 2, 4, 2, 1),  # 64
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            sn_conv2d(ndf * 2, ndf * 4, 4, 2, 1),  # 32
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            sn_conv2d(ndf * 4, ndf * 8, 4, 2, 1),  # 16
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            sn_conv2d(ndf * 8, ndf * 16, 4, 2, 1),  # 8
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),

            sn_conv2d(ndf * 16, 1, 8, 1, 0),  # 1x1
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        label_emb = self.label_embedding(labels).view(-1, 1, 256, 256)
        x = torch.cat([img, label_emb], dim=1)
        return self.main(x).view(-1)


# ----------------------------
# 2. 改进后的条件生成器 (Upsample + Conv)
# ----------------------------
class ConditionalGenerator(nn.Module):
    def __init__(self, nz=100, ngf=32, num_classes=6):
        super().__init__()
        self.nz = nz
        self.num_classes = num_classes
        self.label_embedding = nn.Embedding(num_classes, nz)
        self.ngf = ngf

        self.l1 = nn.Sequential(
            nn.Linear(nz * 2, ngf * 16 * 4 * 4, bias=False),
            nn.BatchNorm1d(ngf * 16 * 4 * 4),
            nn.ReLU(True)
        )

        def up_block(in_c, out_c):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_c, out_c, 3, 1, 1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(True)
            )

        self.up1 = up_block(ngf * 16, ngf * 8)  # 4 -> 8
        self.up2 = up_block(ngf * 8, ngf * 4)  # 8 -> 16
        self.up3 = up_block(ngf * 4, ngf * 2)  # 16 -> 32
        self.up4 = up_block(ngf * 2, ngf)  # 32 -> 64
        self.up5 = up_block(ngf, ngf // 2)  # 64 -> 128

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(ngf // 2, 1, 3, 1, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        noise = noise.view(-1, self.nz)
        label_emb = self.label_embedding(labels)
        x = torch.cat([noise, label_emb], dim=1)
        x = self.l1(x)
        x = x.view(-1, self.ngf * 16, 4, 4)

        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.up5(x)
        return self.final(x)


# ----------------------------
# 3. 数据集
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
        try:
            img = cv2.imread(str(self.img_paths[idx]), cv2.IMREAD_GRAYSCALE)
            if img is None: return self.__getitem__((idx + 1) % len(self))
            img = cv2.resize(img, (self.size, self.size))
            img = (img.astype(np.float32) / 255.0 - 0.5) * 2.0
            img = np.expand_dims(img, axis=0)
            label = self.labels[idx]
            return torch.from_numpy(img), torch.tensor(label, dtype=torch.long)
        except Exception:
            return self.__getitem__((idx + 1) % len(self))


# ----------------------------
# 4. 训练函数 (含断点续训 + CSV日志)
# ----------------------------
def train_cgan_256(
        data_dir,
        output_dir,
        epochs=500,
        batch_size=4,
        nz=100,
        lr=0.0001,
        save_interval=50,
        num_test_samples=1,
        resume=True
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 使用设备: {device}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    dataset = DefectDataset(data_dir, size=256)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    num_classes = len(dataset.class_names)

    netG = ConditionalGenerator(nz=nz, ngf=32, num_classes=num_classes).to(device)
    netD = ConditionalDiscriminator(ndf=32, num_classes=num_classes).to(device)

    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    fixed_noise = torch.randn(num_classes * num_test_samples, nz, 1, 1, device=device)
    fixed_labels = torch.arange(num_classes).repeat_interleave(num_test_samples).to(device)

    # === 断点加载逻辑 ===
    start_epoch = 1
    checkpoint_path = os.path.join(output_dir, "checkpoint_latest.pth")

    if resume and os.path.exists(checkpoint_path):
        print(f"🔄 发现断点，正在恢复: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        netG.load_state_dict(checkpoint['netG_state'])
        netD.load_state_dict(checkpoint['netD_state'])
        optimizerG.load_state_dict(checkpoint['optimizerG_state'])
        optimizerD.load_state_dict(checkpoint['optimizerD_state'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"✅ 已恢复至 Epoch {start_epoch}")
    else:
        print("🆕 从头开始训练")

        def weights_init(m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif classname.find('BatchNorm') != -1:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

        netG.apply(weights_init)
        netD.apply(weights_init)

    # === CSV 日志初始化 ===
    log_csv_path = os.path.join(output_dir, "training_log.csv")
    # 如果是从头训练（或者文件不存在），写入表头
    if start_epoch == 1 or not os.path.exists(log_csv_path):
        with open(log_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'D_loss', 'G_loss'])

    # 训练循环
    for epoch in range(start_epoch, epochs + 1):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")
        total_d_loss = 0
        total_g_loss = 0

        for real_imgs, labels in pbar:
            real_imgs = real_imgs.to(device)
            labels = labels.to(device)
            b_size = real_imgs.size(0)

            # 更新 D
            netD.zero_grad()
            label_real = torch.full((b_size,), 0.9, device=device)
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

            d_loss = errD_real.item() + errD_fake.item()
            total_d_loss += d_loss

            # 更新 G
            netG.zero_grad()
            label_g = torch.ones(b_size, device=device)
            output = netD(fake, labels)
            errG = criterion(output, label_g)
            errG.backward()
            optimizerG.step()

            total_g_loss += errG.item()

            pbar.set_postfix(D_loss=d_loss, G_loss=errG.item())

        # === 写入 CSV 日志 ===
        avg_d_loss = total_d_loss / len(dataloader)
        avg_g_loss = total_g_loss / len(dataloader)
        with open(log_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, avg_d_loss, avg_g_loss])

        # === 保存最新断点 ===
        checkpoint = {
            'epoch': epoch,
            'netG_state': netG.state_dict(),
            'netD_state': netD.state_dict(),
            'optimizerG_state': optimizerG.state_dict(),
            'optimizerD_state': optimizerD.state_dict()
        }
        torch.save(checkpoint, checkpoint_path)

        # === 定期保存可视化样本 ===
        if epoch % save_interval == 0 or epoch == epochs:
            with torch.no_grad():
                netG.eval()
                fake_imgs = netG(fixed_noise, fixed_labels).cpu()
                netG.train()
                for i in range(num_classes):
                    img = fake_imgs[i].squeeze().numpy()
                    img = (img * 0.5 + 0.5) * 255
                    img = np.clip(img, 0, 255).astype(np.uint8)
                    out_path = Path(output_dir) / f"epoch_{epoch:04d}_class_{dataset.class_names[i]}.png"
                    cv2.imwrite(str(out_path), img)
            print(f"🖼️  已保存第 {epoch} 轮样本")

    print(f"✅ 训练完成！结果保存至: {output_dir}")