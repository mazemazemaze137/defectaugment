import csv
import math
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def _validate_image_size(image_size):
    if image_size not in (128, 256):
        raise ValueError(f"image_size must be 128 or 256, got {image_size}")


class ConditionalDiscriminator(nn.Module):
    def __init__(self, ndf=32, num_classes=6, image_size=256):
        super().__init__()
        _validate_image_size(image_size)
        self.image_size = image_size
        self.label_embedding = nn.Embedding(num_classes, image_size * image_size)

        def sn_conv2d(in_c, out_c, k, s, p):
            return nn.utils.spectral_norm(
                nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=False)
            )

        num_downsamples = int(math.log2(image_size)) - 2
        layers = []
        in_channels = 2
        for i in range(num_downsamples):
            out_channels = ndf * min(2 ** i, 8)
            layers.append(sn_conv2d(in_channels, out_channels, 4, 2, 1))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_channels = out_channels

        layers.append(sn_conv2d(in_channels, 1, 4, 1, 0))
        self.main = nn.Sequential(*layers)

    def forward(self, img, labels):
        label_map = self.label_embedding(labels).view(-1, 1, self.image_size, self.image_size)
        x = torch.cat([img, label_map], dim=1)
        return self.main(x).view(-1)


class ConditionalGenerator(nn.Module):
    def __init__(self, nz=100, ngf=64, num_classes=6, image_size=256):
        super().__init__()
        _validate_image_size(image_size)
        self.nz = nz
        self.image_size = image_size
        self.label_embedding = nn.Embedding(num_classes, nz)

        self.start_channels = ngf * 8
        self.fc = nn.Sequential(
            nn.Linear(nz * 2, self.start_channels * 4 * 4, bias=False),
            nn.BatchNorm1d(self.start_channels * 4 * 4),
            nn.ReLU(True),
        )

        num_upsamples = int(math.log2(image_size)) - 2
        blocks = []
        in_channels = self.start_channels
        for _ in range(num_upsamples - 1):
            out_channels = max(ngf // 2, in_channels // 2)
            blocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(True),
                )
            )
            in_channels = out_channels
        self.blocks = nn.ModuleList(blocks)
        self.final = nn.ConvTranspose2d(in_channels, 1, 4, 2, 1, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, noise, labels):
        noise = noise.view(-1, self.nz)
        label_emb = self.label_embedding(labels)
        x = torch.cat([noise, label_emb], dim=1)
        x = self.fc(x).view(-1, self.start_channels, 4, 4)
        if not self.blocks:
            raise RuntimeError("Generator upsampling blocks are empty.")
        for block in self.blocks:
            x = block(x)
        x = self.final(x)
        return self.tanh(x)


class DefectDataset(Dataset):
    def __init__(self, root_dir, size=256):
        self.root_dir = Path(root_dir)
        self.size = size
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Directory not found: {root_dir}")

        self.class_names = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        if not self.class_names:
            raise ValueError(f"No class subdirectories found in {root_dir}")

        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}
        self.img_paths = []
        self.labels = []

        for class_name in self.class_names:
            class_dir = self.root_dir / class_name
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff"):
                for p in class_dir.glob(ext):
                    self.img_paths.append(p)
                    self.labels.append(self.class_to_idx[class_name])

        if not self.img_paths:
            raise ValueError(f"No images found in {root_dir}")
        print(f"Loaded {len(self.img_paths)} images across {len(self.class_names)} classes.")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        try:
            img = cv2.imread(str(self.img_paths[idx]), cv2.IMREAD_GRAYSCALE)
            if img is None:
                return self.__getitem__((idx + 1) % len(self))
            img = cv2.resize(img, (self.size, self.size))
            img = (img.astype(np.float32) / 255.0 - 0.5) * 2.0
            img = np.expand_dims(img, axis=0)
            label = self.labels[idx]
            return torch.from_numpy(img), torch.tensor(label, dtype=torch.long)
        except Exception:
            return self.__getitem__((idx + 1) % len(self))


def train_cgan_256(
    data_dir,
    output_dir,
    epochs=500,
    batch_size=4,
    nz=100,
    lr=0.0002,
    lr_g=None,
    lr_d=None,
    image_size=128,
    save_interval=20,
    num_test_samples=8,
    resume=True,
    pause_event=None,
    stop_event=None,
    status_callback=None,
):
    _validate_image_size(image_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Training image size: {image_size}x{image_size}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    dataset = DefectDataset(data_dir, size=image_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    num_classes = len(dataset.class_names)

    net_g = ConditionalGenerator(nz=nz, ngf=64, num_classes=num_classes, image_size=image_size).to(device)
    net_d = ConditionalDiscriminator(ndf=32, num_classes=num_classes, image_size=image_size).to(device)

    g_lr = lr if lr_g is None else lr_g
    d_lr = lr if lr_d is None else lr_d
    optimizer_g = optim.Adam(net_g.parameters(), lr=g_lr, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(net_d.parameters(), lr=d_lr, betas=(0.5, 0.999))
    criterion = nn.BCEWithLogitsLoss()

    fixed_noise = torch.randn(num_classes * num_test_samples, nz, 1, 1, device=device)
    fixed_labels = torch.arange(num_classes, device=device).repeat_interleave(num_test_samples)

    start_epoch = 1
    checkpoint_path = os.path.join(output_dir, "checkpoint_latest.pth")

    if resume and os.path.exists(checkpoint_path):
        print(f"Found checkpoint, resuming: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        try:
            net_g.load_state_dict(checkpoint["netG_state"])
            net_d.load_state_dict(checkpoint["netD_state"])
            optimizer_g.load_state_dict(checkpoint["optimizerG_state"])
            optimizer_d.load_state_dict(checkpoint["optimizerD_state"])
            start_epoch = checkpoint["epoch"] + 1
            print(f"Resumed from epoch {start_epoch}")
        except RuntimeError as exc:
            print(f"Checkpoint incompatible with current model settings, restarting: {exc}")
            start_epoch = 1
    else:
        print("Training from scratch.")

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1 or classname.find("Linear") != -1:
            if hasattr(m, "weight") and m.weight is not None:
                nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    if start_epoch == 1:
        net_g.apply(weights_init)
        net_d.apply(weights_init)

    log_csv_path = os.path.join(output_dir, "training_log.csv")
    if start_epoch == 1 or not os.path.exists(log_csv_path):
        with open(log_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "D_loss", "G_loss"])

    def save_checkpoint(epoch):
        checkpoint = {
            "epoch": epoch,
            "netG_state": net_g.state_dict(),
            "netD_state": net_d.state_dict(),
            "optimizerG_state": optimizer_g.state_dict(),
            "optimizerD_state": optimizer_d.state_dict(),
            "image_size": image_size,
            "nz": nz,
            "num_classes": num_classes,
        }
        torch.save(checkpoint, checkpoint_path)

    stopped_early = False
    last_finished_epoch = start_epoch - 1

    for epoch in range(start_epoch, epochs + 1):
        if stop_event is not None and stop_event.is_set():
            stopped_early = True
            break

        if status_callback is not None:
            status_callback({"state": "running", "epoch": epoch, "epochs": epochs})

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")
        total_d_loss = 0.0
        total_g_loss = 0.0

        for real_imgs, labels in pbar:
            while pause_event is not None and pause_event.is_set():
                if status_callback is not None:
                    status_callback({"state": "paused", "epoch": epoch, "epochs": epochs})
                if stop_event is not None and stop_event.is_set():
                    stopped_early = True
                    break
                time.sleep(0.2)
            if stopped_early:
                break

            if stop_event is not None and stop_event.is_set():
                stopped_early = True
                break

            real_imgs = real_imgs.to(device)
            labels = labels.to(device)
            b_size = real_imgs.size(0)

            net_d.zero_grad(set_to_none=True)
            label_real = torch.full((b_size,), 0.9, device=device)
            pred_real = net_d(real_imgs, labels)
            err_d_real = criterion(pred_real, label_real)

            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake = net_g(noise, labels)
            label_fake = torch.zeros(b_size, device=device)
            pred_fake = net_d(fake.detach(), labels)
            err_d_fake = criterion(pred_fake, label_fake)

            err_d = err_d_real + err_d_fake
            err_d.backward()
            optimizer_d.step()
            total_d_loss += err_d.item()

            net_g.zero_grad(set_to_none=True)
            label_g = torch.ones(b_size, device=device)
            pred_g = net_d(fake, labels)
            err_g = criterion(pred_g, label_g)
            err_g.backward()
            optimizer_g.step()
            total_g_loss += err_g.item()

            pbar.set_postfix(D_loss=err_d.item(), G_loss=err_g.item())

        if stopped_early:
            break

        avg_d_loss = total_d_loss / len(dataloader)
        avg_g_loss = total_g_loss / len(dataloader)
        with open(log_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, avg_d_loss, avg_g_loss])

        save_checkpoint(epoch)
        last_finished_epoch = epoch

        if epoch % save_interval == 0 or epoch == epochs:
            with torch.no_grad():
                net_g.eval()
                fake_imgs = net_g(fixed_noise, fixed_labels).cpu()
                net_g.train()

            for class_idx, class_name in enumerate(dataset.class_names):
                for sample_idx in range(num_test_samples):
                    tensor_idx = class_idx * num_test_samples + sample_idx
                    img = fake_imgs[tensor_idx].squeeze().numpy()
                    img = (img * 0.5 + 0.5) * 255.0
                    img = np.clip(img, 0, 255).astype(np.uint8)
                    if num_test_samples == 1:
                        out_name = f"epoch_{epoch:04d}_class_{class_name}.png"
                    else:
                        out_name = f"epoch_{epoch:04d}_class_{class_name}_s{sample_idx:02d}.png"
                    out_path = Path(output_dir) / out_name
                    cv2.imwrite(str(out_path), img)
            print(f"Saved samples for epoch {epoch}")

    if stopped_early:
        if last_finished_epoch >= start_epoch:
            save_checkpoint(last_finished_epoch)
        if status_callback is not None:
            status_callback({"state": "stopped", "epoch": last_finished_epoch, "epochs": epochs})
        print(f"Training stopped early at epoch {last_finished_epoch}. Results saved to: {output_dir}")
        return {"status": "stopped", "last_epoch": last_finished_epoch, "output_dir": output_dir}

    if status_callback is not None:
        status_callback({"state": "finished", "epoch": epochs, "epochs": epochs})
    print(f"Training finished. Results saved to: {output_dir}")
    return {"status": "finished", "last_epoch": epochs, "output_dir": output_dir}
