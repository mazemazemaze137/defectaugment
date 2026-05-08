import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _image_stats(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    mean = float(np.mean(img))
    std = float(np.std(img))
    sharpness = float(cv2.Laplacian(img, cv2.CV_64F).var())
    return {"mean": mean, "std": std, "sharpness": sharpness}


def filter_generated_samples(
    input_dir,
    output_dir,
    max_per_class=50,
    min_sharpness=5.0,
    min_std=8.0,
    min_mean=15.0,
    max_mean=240.0,
):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "max_per_class": max_per_class,
        "min_sharpness": min_sharpness,
        "min_std": min_std,
        "min_mean": min_mean,
        "max_mean": max_mean,
        "classes": {},
    }

    for class_dir in sorted([p for p in input_dir.iterdir() if p.is_dir()]):
        candidates = []
        total = 0
        for path in sorted(class_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            total += 1
            stats = _image_stats(path)
            if stats is None:
                continue
            if stats["sharpness"] < min_sharpness:
                continue
            if stats["std"] < min_std:
                continue
            if not (min_mean <= stats["mean"] <= max_mean):
                continue
            candidates.append((path, stats))

        candidates.sort(key=lambda item: (item[1]["sharpness"], item[1]["std"]), reverse=True)
        selected = candidates[:max_per_class]
        out_class_dir = output_dir / class_dir.name
        out_class_dir.mkdir(parents=True, exist_ok=True)
        for src_path, _ in selected:
            shutil.copy2(src_path, out_class_dir / src_path.name)

        summary["classes"][class_dir.name] = {
            "total": total,
            "passed_filters": len(candidates),
            "selected": len(selected),
            "avg_selected_sharpness": float(np.mean([s["sharpness"] for _, s in selected])) if selected else 0.0,
            "avg_selected_std": float(np.mean([s["std"] for _, s in selected])) if selected else 0.0,
        }

    summary["total_selected"] = sum(item["selected"] for item in summary["classes"].values())
    (output_dir / "filter_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Filter generated defect samples by sharpness and grayscale statistics.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-per-class", type=int, default=50)
    parser.add_argument("--min-sharpness", type=float, default=5.0)
    parser.add_argument("--min-std", type=float, default=8.0)
    parser.add_argument("--min-mean", type=float, default=15.0)
    parser.add_argument("--max-mean", type=float, default=240.0)
    return parser.parse_args()


def main():
    args = parse_args()
    summary = filter_generated_samples(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        max_per_class=args.max_per_class,
        min_sharpness=args.min_sharpness,
        min_std=args.min_std,
        min_mean=args.min_mean,
        max_mean=args.max_mean,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
