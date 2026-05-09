import argparse
import csv
import json
import random
import re
import shutil
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
GENERATED_CLASS_PATTERN = re.compile(r"epoch_\d+_class_(.+?)(?:_s\d+)?\.(?:png|jpg|jpeg|bmp|tif|tiff)$", re.I)


def _normalize_class_name(name):
    return name.strip().lower().replace("_", "-")


def _collect_folder_images(root_dir, class_to_idx=None):
    root_dir = Path(root_dir)
    if not root_dir.exists():
        raise FileNotFoundError(f"Directory not found: {root_dir}")

    class_dirs = sorted([p for p in root_dir.iterdir() if p.is_dir()])
    if class_to_idx is None:
        class_names = [_normalize_class_name(p.name) for p in class_dirs]
        class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    samples = []
    for class_dir in class_dirs:
        class_name = _normalize_class_name(class_dir.name)
        if class_name not in class_to_idx:
            continue
        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append((path, class_to_idx[class_name]))
    return samples, class_to_idx


def _collect_generated_images(generated_dir, class_to_idx):
    if not generated_dir:
        return []

    generated_dir = Path(generated_dir)
    if not generated_dir.exists():
        raise FileNotFoundError(f"Generated directory not found: {generated_dir}")

    samples = []
    for path in sorted(generated_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        parent_class = _normalize_class_name(path.parent.name)
        if parent_class in class_to_idx:
            samples.append((path, class_to_idx[parent_class]))
            continue

        match = GENERATED_CLASS_PATTERN.match(path.name)
        if match:
            class_name = _normalize_class_name(match.group(1))
            if class_name in class_to_idx:
                samples.append((path, class_to_idx[class_name]))
    return samples


class DefectClassificationDataset(Dataset):
    def __init__(self, samples, image_size=128, include_path=False):
        if not samples:
            raise ValueError("No image samples found.")
        self.samples = samples
        self.image_size = image_size
        self.include_path = include_path

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Failed to read image: {path}")
        img = cv2.resize(img, (self.image_size, self.image_size))
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        img = np.expand_dims(img, axis=0)
        if self.include_path:
            return torch.from_numpy(img), torch.tensor(label, dtype=torch.long), str(path)
        return torch.from_numpy(img), torch.tensor(label, dtype=torch.long)


class SmallDefectCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def build_classifier_model(model_name, num_classes):
    model_name = (model_name or "small_cnn").lower()
    if model_name == "small_cnn":
        return SmallDefectCNN(num_classes=num_classes)
    if model_name == "resnet18":
        model = models.resnet18(weights=None)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if model_name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=None)
        first_conv = model.features[0][0]
        model.features[0][0] = nn.Conv2d(
            1,
            first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=False,
        )
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    raise ValueError(f"Unsupported classifier model: {model_name}")


def _build_loaders(train_dir, val_dir, generated_dir=None, image_size=128, batch_size=32, num_workers=0):
    train_samples, class_to_idx = _collect_folder_images(train_dir)
    generated_samples = _collect_generated_images(generated_dir, class_to_idx)
    val_samples, _ = _collect_folder_images(val_dir, class_to_idx=class_to_idx)

    train_dataset = DefectClassificationDataset(train_samples + generated_samples, image_size=image_size)
    val_dataset = DefectClassificationDataset(val_samples, image_size=image_size)
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    counts = {
        "train_original": len(train_samples),
        "train_generated": len(generated_samples),
        "train_total": len(train_dataset),
        "validation": len(val_dataset),
    }
    return train_loader, val_loader, class_names, counts, val_samples


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def parse_class_weights(text):
    weights = {}
    if not text:
        return weights
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid class weight item: {item}. Expected class=value")
        class_name, value = item.split("=", 1)
        weights[_normalize_class_name(class_name)] = float(value)
    return weights


def _build_class_weight_tensor(class_names, class_weights, device):
    if not class_weights:
        return None, {name: 1.0 for name in class_names}
    resolved = []
    summary = {}
    for class_name in class_names:
        weight = float(class_weights.get(_normalize_class_name(class_name), 1.0))
        resolved.append(weight)
        summary[class_name] = weight
    return torch.tensor(resolved, dtype=torch.float32, device=device), summary


_parse_class_weights = parse_class_weights


def _evaluate(model, loader, criterion, device, num_classes):
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            preds = outputs.argmax(dim=1)

            total_loss += loss.item() * labels.size(0)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            for true_label, pred_label in zip(labels.cpu().numpy(), preds.cpu().numpy()):
                confusion[int(true_label), int(pred_label)] += 1

    return {
        "loss": total_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
        "confusion_matrix": confusion,
    }


def _save_confusion_matrix(confusion, class_names, output_path):
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(confusion, cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(class_names)), labels=class_names, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(class_names)), labels=class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")

    threshold = confusion.max() / 2 if confusion.size and confusion.max() > 0 else 0
    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            ax.text(
                j,
                i,
                str(confusion[i, j]),
                ha="center",
                va="center",
                color="white" if confusion[i, j] > threshold else "black",
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _save_history_plot(history, output_path):
    if not history:
        return
    epochs = [row["epoch"] for row in history]
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(epochs, [row["train_accuracy"] for row in history], label="train_acc", color="#2563eb")
    ax1.plot(epochs, [row["val_accuracy"] for row in history], label="val_acc", color="#16a34a")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(0, 1.05)
    ax2 = ax1.twinx()
    ax2.plot(epochs, [row["train_loss"] for row in history], label="train_loss", color="#f97316", linestyle="--")
    ax2.plot(epochs, [row["val_loss"] for row in history], label="val_loss", color="#dc2626", linestyle="--")
    ax2.set_ylabel("Loss")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _export_low_confidence_samples(
    model,
    samples,
    class_names,
    output_dir,
    image_size,
    device,
    threshold=0.7,
    max_samples=80,
    include_errors=True,
):
    if threshold <= 0 and not include_errors:
        return {"csv_path": "", "image_dir": "", "num_records": 0}

    dataset = DefectClassificationDataset(samples, image_size=image_size, include_path=True)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())
    records = []
    model.eval()
    with torch.no_grad():
        for images, labels, paths in loader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            confidences, preds = probs.max(dim=1)
            for path, true_label, pred_label, confidence in zip(
                paths,
                labels.cpu().numpy(),
                preds.cpu().numpy(),
                confidences.cpu().numpy(),
            ):
                correct = int(true_label) == int(pred_label)
                if float(confidence) <= threshold or (include_errors and not correct):
                    records.append(
                        {
                            "path": path,
                            "true_label": class_names[int(true_label)],
                            "pred_label": class_names[int(pred_label)],
                            "confidence": float(confidence),
                            "correct": correct,
                        }
                    )

    records.sort(key=lambda row: (row["correct"], row["confidence"]))
    selected = records[:max_samples]
    image_dir = output_dir / "low_confidence_samples"
    image_dir.mkdir(parents=True, exist_ok=True)
    for idx, row in enumerate(selected, start=1):
        source = Path(row["path"])
        safe_name = f"{idx:03d}_true-{row['true_label']}_pred-{row['pred_label']}_conf-{row['confidence']:.3f}{source.suffix.lower()}"
        target = image_dir / safe_name
        try:
            shutil.copy2(source, target)
            row["exported_path"] = str(target)
        except OSError:
            row["exported_path"] = ""

    csv_path = output_dir / "low_confidence_samples.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["path", "true_label", "pred_label", "confidence", "correct", "exported_path"],
        )
        writer.writeheader()
        writer.writerows(selected)
    return {"csv_path": str(csv_path), "image_dir": str(image_dir), "num_records": len(selected)}


def run_classification_validation(
    train_dir,
    val_dir,
    output_dir,
    generated_dir=None,
    image_size=128,
    epochs=5,
    batch_size=32,
    lr=1e-3,
    num_workers=0,
    seed=42,
    amp=True,
    early_stopping_patience=0,
    class_weights=None,
    quiet=False,
    model_name="small_cnn",
    low_confidence_threshold=0.7,
    max_low_confidence_samples=80,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, class_names, counts, val_samples = _build_loaders(
        train_dir=train_dir,
        val_dir=val_dir,
        generated_dir=generated_dir,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    model_name = (model_name or "small_cnn").lower()
    model = build_classifier_model(model_name, num_classes=len(class_names)).to(device)
    weight_tensor, resolved_class_weights = _build_class_weight_tensor(class_names, class_weights or {}, device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    use_amp = bool(amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history = []
    start_time = time.perf_counter()
    best_accuracy = 0.0
    best_epoch = 0
    best_state = None
    final_eval = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Classifier epoch {epoch}/{epochs}", disable=quiet)
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            preds = outputs.argmax(dim=1)
            train_loss += loss.item() * labels.size(0)
            train_total += labels.size(0)
            train_correct += (preds == labels).sum().item()
            pbar.set_postfix(loss=loss.item())

        train_loss /= max(train_total, 1)
        train_accuracy = train_correct / max(train_total, 1)
        val_eval = _evaluate(model, val_loader, criterion, device, len(class_names))
        final_eval = val_eval

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_eval["loss"],
            "val_accuracy": val_eval["accuracy"],
        }
        history.append(row)

        if val_eval["accuracy"] >= best_accuracy:
            best_accuracy = val_eval["accuracy"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if early_stopping_patience > 0 and patience_counter >= early_stopping_patience:
                if not quiet:
                    print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
                break

    elapsed_seconds = time.perf_counter() - start_time

    if best_state is not None:
        torch.save(best_state, output_dir / "classifier_best.pth")
        model.load_state_dict(best_state)
        best_eval = _evaluate(model, val_loader, criterion, device, len(class_names))
    else:
        best_eval = final_eval

    with open(output_dir / "history.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy"],
        )
        writer.writeheader()
        writer.writerows(history)
    _save_history_plot(history, output_dir / "history.png")

    confusion = best_eval["confusion_matrix"] if best_eval is not None else np.zeros((len(class_names), len(class_names)))
    np.savetxt(output_dir / "confusion_matrix.csv", confusion, fmt="%d", delimiter=",")
    _save_confusion_matrix(confusion, class_names, output_dir / "confusion_matrix.png")
    low_confidence = _export_low_confidence_samples(
        model=model,
        samples=val_samples,
        class_names=class_names,
        output_dir=output_dir,
        image_size=image_size,
        device=device,
        threshold=low_confidence_threshold,
        max_samples=max_low_confidence_samples,
        include_errors=True,
    )

    summary = {
        "device": str(device),
        "model_name": model_name,
        "class_names": class_names,
        "counts": counts,
        "epochs": epochs,
        "completed_epochs": len(history),
        "best_epoch": best_epoch,
        "batch_size": batch_size,
        "image_size": image_size,
        "lr": lr,
        "seed": seed,
        "amp": use_amp,
        "elapsed_seconds": elapsed_seconds,
        "best_val_accuracy": best_accuracy,
        "final_val_accuracy": history[-1]["val_accuracy"] if history else 0.0,
        "best_val_loss": best_eval["loss"] if best_eval is not None else 0.0,
        "early_stopping_patience": early_stopping_patience,
        "class_weights": resolved_class_weights,
        "low_confidence_threshold": low_confidence_threshold,
        "low_confidence": low_confidence,
        "output_dir": str(output_dir),
        "generated_dir": str(generated_dir) if generated_dir else "",
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run downstream classification validation for defect augmentation.")
    parser.add_argument("--train-dir", default="data/raw/NEU-DET/train/images")
    parser.add_argument("--val-dir", default="data/raw/NEU-DET/validation/images")
    parser.add_argument("--generated-dir", default="")
    parser.add_argument("--output-dir", default="results/classifier_validation")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument(
        "--class-weights",
        default="",
        help="Optional comma-separated class loss weights, e.g. pitted-surface=1.5,crazing=1.2",
    )
    parser.add_argument("--quiet", action="store_true", help="Disable per-batch progress bars.")
    parser.add_argument("--model-name", default="small_cnn", choices=["small_cnn", "resnet18", "mobilenet_v3_small"])
    parser.add_argument("--low-confidence-threshold", type=float, default=0.7)
    parser.add_argument("--max-low-confidence-samples", type=int, default=80)
    args = parser.parse_args()

    summary = run_classification_validation(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        generated_dir=args.generated_dir or None,
        output_dir=args.output_dir,
        image_size=args.image_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        seed=args.seed,
        amp=not args.no_amp,
        early_stopping_patience=args.early_stopping_patience,
        class_weights=parse_class_weights(args.class_weights),
        quiet=args.quiet,
        model_name=args.model_name,
        low_confidence_threshold=args.low_confidence_threshold,
        max_low_confidence_samples=args.max_low_confidence_samples,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
