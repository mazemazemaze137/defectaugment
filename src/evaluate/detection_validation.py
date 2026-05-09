import argparse
import csv
import json
import random
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.ops import box_iou
from tqdm import tqdm


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")


def _normalize_class_name(name):
    return name.strip().lower().replace("_", "-")


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_image_path(images_root, filename):
    images_root = Path(images_root)
    direct = images_root / filename
    if direct.exists():
        return direct
    matches = list(images_root.rglob(filename))
    if matches:
        return matches[0]
    stem = Path(filename).stem
    for suffix in IMAGE_SUFFIXES:
        matches = list(images_root.rglob(stem + suffix))
        if matches:
            return matches[0]
    return None


def parse_voc_annotations(images_root, annotations_dir, class_to_idx=None, max_samples=None):
    images_root = Path(images_root)
    annotations_dir = Path(annotations_dir)
    if not images_root.exists():
        raise FileNotFoundError(f"Image root not found: {images_root}")
    if not annotations_dir.exists():
        raise FileNotFoundError(f"Annotation directory not found: {annotations_dir}")

    records = []
    discovered_classes = set()
    for xml_path in sorted(annotations_dir.glob("*.xml")):
        root = ET.parse(xml_path).getroot()
        filename = root.findtext("filename", "")
        image_path = _resolve_image_path(images_root, filename)
        if image_path is None:
            continue

        boxes = []
        labels = []
        for obj in root.findall("object"):
            class_name = _normalize_class_name(obj.findtext("name", "unknown"))
            discovered_classes.add(class_name)
            box = obj.find("bndbox")
            if box is None:
                continue
            xmin = float(box.findtext("xmin", "0"))
            ymin = float(box.findtext("ymin", "0"))
            xmax = float(box.findtext("xmax", "0"))
            ymax = float(box.findtext("ymax", "0"))
            if xmax <= xmin or ymax <= ymin:
                continue
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(class_name)

        if boxes:
            records.append({"image_path": image_path, "boxes": boxes, "labels": labels})
        if max_samples and len(records) >= max_samples:
            break

    if class_to_idx is None:
        class_names = sorted(discovered_classes)
        class_to_idx = {name: idx + 1 for idx, name in enumerate(class_names)}

    filtered = []
    for record in records:
        numeric_labels = [class_to_idx[_normalize_class_name(label)] for label in record["labels"] if _normalize_class_name(label) in class_to_idx]
        if len(numeric_labels) != len(record["boxes"]):
            continue
        filtered.append({**record, "numeric_labels": numeric_labels})
    return filtered, class_to_idx


class VocDetectionDataset(Dataset):
    def __init__(self, records, image_size=320):
        self.records = records
        self.image_size = image_size

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image = cv2.imread(str(record["image_path"]), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Cannot read image: {record['image_path']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        scale_x = self.image_size / max(orig_w, 1)
        scale_y = self.image_size / max(orig_h, 1)
        boxes = np.array(record["boxes"], dtype=np.float32)
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y
        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(record["numeric_labels"], dtype=torch.int64),
            "image_id": torch.tensor([index], dtype=torch.int64),
        }
        image_tensor = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1)
        return image_tensor, target


def _collate(batch):
    return tuple(zip(*batch))


def _match_predictions(prediction, target, num_classes, score_threshold=0.3, iou_threshold=0.5):
    stats = {class_idx: {"scores": [], "tp": [], "num_gt": 0} for class_idx in range(1, num_classes)}
    gt_boxes = target["boxes"].cpu()
    gt_labels = target["labels"].cpu()
    pred_boxes = prediction["boxes"].detach().cpu()
    pred_labels = prediction["labels"].detach().cpu()
    pred_scores = prediction["scores"].detach().cpu()

    for class_idx in range(1, num_classes):
        class_gt_mask = gt_labels == class_idx
        class_pred_mask = (pred_labels == class_idx) & (pred_scores >= score_threshold)
        class_gt_boxes = gt_boxes[class_gt_mask]
        class_pred_boxes = pred_boxes[class_pred_mask]
        class_scores = pred_scores[class_pred_mask]
        stats[class_idx]["num_gt"] += int(class_gt_boxes.shape[0])
        if class_pred_boxes.numel() == 0:
            continue

        order = torch.argsort(class_scores, descending=True)
        class_pred_boxes = class_pred_boxes[order]
        class_scores = class_scores[order]
        matched = set()
        ious = box_iou(class_pred_boxes, class_gt_boxes) if class_gt_boxes.numel() else torch.zeros((len(class_pred_boxes), 0))
        for pred_idx, score in enumerate(class_scores):
            is_tp = 0
            if ious.shape[1] > 0:
                best_iou, best_gt = torch.max(ious[pred_idx], dim=0)
                best_gt_int = int(best_gt.item())
                if float(best_iou) >= iou_threshold and best_gt_int not in matched:
                    matched.add(best_gt_int)
                    is_tp = 1
            stats[class_idx]["scores"].append(float(score))
            stats[class_idx]["tp"].append(is_tp)
    return stats


def _merge_stats(total, update):
    for class_idx, row in update.items():
        total[class_idx]["scores"].extend(row["scores"])
        total[class_idx]["tp"].extend(row["tp"])
        total[class_idx]["num_gt"] += row["num_gt"]


def _average_precision(scores, tp, num_gt):
    if num_gt <= 0:
        return None
    if not scores:
        return 0.0, 0.0, 0.0
    order = np.argsort(-np.array(scores))
    tp_sorted = np.array(tp)[order]
    fp_sorted = 1 - tp_sorted
    cum_tp = np.cumsum(tp_sorted)
    cum_fp = np.cumsum(fp_sorted)
    recalls = cum_tp / max(num_gt, 1)
    precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1)
    ap = 0.0
    for threshold in np.linspace(0, 1, 11):
        precision_at_recall = precisions[recalls >= threshold]
        ap += (precision_at_recall.max() if precision_at_recall.size else 0.0) / 11.0
    return float(ap), float(precisions[-1]), float(recalls[-1])


def evaluate_detector(model, loader, device, class_names, score_threshold=0.3, iou_threshold=0.5):
    model.eval()
    stats = {idx: {"scores": [], "tp": [], "num_gt": 0} for idx in range(1, len(class_names) + 1)}
    with torch.no_grad():
        for images, targets in loader:
            images = [image.to(device) for image in images]
            predictions = model(images)
            for prediction, target in zip(predictions, targets):
                update = _match_predictions(
                    prediction,
                    target,
                    num_classes=len(class_names) + 1,
                    score_threshold=score_threshold,
                    iou_threshold=iou_threshold,
                )
                _merge_stats(stats, update)

    rows = []
    aps = []
    for idx, class_name in enumerate(class_names, start=1):
        result = _average_precision(stats[idx]["scores"], stats[idx]["tp"], stats[idx]["num_gt"])
        if result is None:
            ap, precision, recall = 0.0, 0.0, 0.0
        else:
            ap, precision, recall = result
            aps.append(ap)
        rows.append(
            {
                "class_name": class_name,
                "ap50": ap,
                "precision": precision,
                "recall": recall,
                "num_gt": stats[idx]["num_gt"],
                "num_predictions": len(stats[idx]["scores"]),
            }
        )
    return {"map50": float(np.mean(aps)) if aps else 0.0, "class_metrics": rows}


def run_detection_validation(
    train_images,
    train_annotations,
    val_images,
    val_annotations,
    output_dir,
    epochs=1,
    batch_size=2,
    image_size=320,
    lr=5e-4,
    max_train=None,
    max_val=None,
    seed=42,
    score_threshold=0.3,
    iou_threshold=0.5,
    quiet=False,
):
    _set_seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_records, class_to_idx = parse_voc_annotations(train_images, train_annotations, max_samples=max_train)
    val_records, _ = parse_voc_annotations(val_images, val_annotations, class_to_idx=class_to_idx, max_samples=max_val)
    if not train_records or not val_records:
        raise ValueError("Detection training/validation records are empty.")

    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    class_names = [idx_to_class[idx] for idx in sorted(idx_to_class)]
    train_loader = DataLoader(
        VocDetectionDataset(train_records, image_size=image_size),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=_collate,
    )
    val_loader = DataLoader(
        VocDetectionDataset(val_records, image_size=image_size),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=_collate,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=None, weights_backbone=None, num_classes=len(class_names) + 1)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    history = []
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_batches = 0
        iterator = tqdm(train_loader, desc=f"Detector epoch {epoch}/{epochs}", disable=quiet)
        for images, targets in iterator:
            images = [image.to(device) for image in images]
            targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
            loss_dict = model(images, targets)
            loss = sum(loss_value for loss_value in loss_dict.values())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            total_batches += 1
            iterator.set_postfix(loss=float(loss.item()))
        history.append({"epoch": epoch, "train_loss": total_loss / max(total_batches, 1)})

    eval_result = evaluate_detector(
        model,
        val_loader,
        device,
        class_names=class_names,
        score_threshold=score_threshold,
        iou_threshold=iou_threshold,
    )
    elapsed = time.perf_counter() - start
    torch.save(model.state_dict(), output_dir / "detector_last.pth")

    with (output_dir / "detection_history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["epoch", "train_loss"])
        writer.writeheader()
        writer.writerows(history)
    with (output_dir / "detection_class_metrics.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=["class_name", "ap50", "precision", "recall", "num_gt", "num_predictions"])
        writer.writeheader()
        writer.writerows(eval_result["class_metrics"])

    summary = {
        "device": str(device),
        "class_names": class_names,
        "train_samples": len(train_records),
        "val_samples": len(val_records),
        "epochs": epochs,
        "batch_size": batch_size,
        "image_size": image_size,
        "lr": lr,
        "seed": seed,
        "score_threshold": score_threshold,
        "iou_threshold": iou_threshold,
        "elapsed_seconds": elapsed,
        "map50": eval_result["map50"],
        "class_metrics": eval_result["class_metrics"],
        "note": "GAN ROI samples are not used here because target detection requires bounding-box annotations.",
        "output_dir": str(output_dir),
    }
    (output_dir / "detection_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run a lightweight VOC-style target detection validation.")
    parser.add_argument("--train-images", default="data/raw/NEU-DET/train/images")
    parser.add_argument("--train-annotations", default="data/raw/NEU-DET/train/annotations")
    parser.add_argument("--val-images", default="data/raw/NEU-DET/validation/images")
    parser.add_argument("--val-annotations", default="data/raw/NEU-DET/validation/annotations")
    parser.add_argument("--output-dir", default="results/detection_validation")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-val", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--score-threshold", type=float, default=0.3)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    summary = run_detection_validation(
        train_images=args.train_images,
        train_annotations=args.train_annotations,
        val_images=args.val_images,
        val_annotations=args.val_annotations,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        lr=args.lr,
        max_train=args.max_train,
        max_val=args.max_val,
        seed=args.seed,
        score_threshold=args.score_threshold,
        iou_threshold=args.iou_threshold,
        quiet=args.quiet,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
