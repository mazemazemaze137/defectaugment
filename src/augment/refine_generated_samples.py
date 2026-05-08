import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _normalize_class_name(name):
    return name.strip().lower().replace("_", "-")


def _collect_class_images(root_dir):
    root_dir = Path(root_dir)
    class_images = {}
    for class_dir in sorted([p for p in root_dir.iterdir() if p.is_dir()]):
        paths = [
            path
            for path in sorted(class_dir.rglob("*"))
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        class_images[_normalize_class_name(class_dir.name)] = (class_dir.name, paths)
    return class_images


def _image_stats(paths):
    means = []
    stds = []
    sharpness = []
    for path in paths:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        means.append(float(np.mean(img)))
        stds.append(float(np.std(img)))
        sharpness.append(float(cv2.Laplacian(img, cv2.CV_64F).var()))
    if not means:
        return {"mean": 127.5, "std": 32.0, "sharpness": 0.0, "count": 0}
    return {
        "mean": float(np.mean(means)),
        "std": float(np.mean(stds)),
        "sharpness": float(np.mean(sharpness)),
        "count": len(means),
    }


def _match_mean_std(img, target_mean, target_std, target_std_scale):
    src_mean = float(np.mean(img))
    src_std = float(np.std(img))
    if src_std < 1e-3:
        return img
    matched = (img.astype(np.float32) - src_mean) / src_std
    matched = matched * max(target_std * target_std_scale, 1.0) + target_mean
    return np.clip(matched, 0, 255).astype(np.uint8)


def _refine_image(img, target_stats, target_std_scale=0.75, clahe_clip=1.5, sharpen_amount=0.25):
    refined = _match_mean_std(
        img,
        target_mean=target_stats["mean"],
        target_std=target_stats["std"],
        target_std_scale=target_std_scale,
    )
    if clahe_clip > 0:
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
        refined = clahe.apply(refined)
    if sharpen_amount > 0:
        blurred = cv2.GaussianBlur(refined, (0, 0), 1.0)
        refined = cv2.addWeighted(refined, 1.0 + sharpen_amount, blurred, -sharpen_amount, 0)
    return refined


def refine_generated_samples(
    generated_dir,
    output_dir,
    real_dir=None,
    classes=None,
    target_std_scale=0.75,
    clahe_clip=1.5,
    sharpen_amount=0.25,
):
    generated_dir = Path(generated_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not generated_dir.exists():
        raise FileNotFoundError(f"Generated directory not found: {generated_dir}")

    selected_classes = {_normalize_class_name(name) for name in classes} if classes else None
    generated_classes = _collect_class_images(generated_dir)
    real_classes = _collect_class_images(real_dir) if real_dir else {}
    real_stats = {
        class_key: _image_stats(paths)
        for class_key, (_, paths) in real_classes.items()
    }

    summary = {
        "generated_dir": str(generated_dir),
        "output_dir": str(output_dir),
        "real_dir": str(real_dir) if real_dir else "",
        "classes": {},
        "target_std_scale": target_std_scale,
        "clahe_clip": clahe_clip,
        "sharpen_amount": sharpen_amount,
    }

    for class_key, (class_name, paths) in generated_classes.items():
        out_class_dir = output_dir / class_name
        out_class_dir.mkdir(parents=True, exist_ok=True)
        should_refine = selected_classes is None or class_key in selected_classes
        target_stats = real_stats.get(class_key)
        before_stats = _image_stats(paths)
        written = 0

        for path in paths:
            out_path = out_class_dir / path.name
            if not should_refine or target_stats is None:
                shutil.copy2(path, out_path)
                written += 1
                continue
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            refined = _refine_image(
                img,
                target_stats=target_stats,
                target_std_scale=target_std_scale,
                clahe_clip=clahe_clip,
                sharpen_amount=sharpen_amount,
            )
            cv2.imwrite(str(out_path), refined)
            written += 1

        after_paths = [
            path
            for path in sorted(out_class_dir.rglob("*"))
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        summary["classes"][class_name] = {
            "refined": bool(should_refine and target_stats is not None),
            "written": written,
            "real_stats": target_stats or {},
            "before_stats": before_stats,
            "after_stats": _image_stats(after_paths),
        }

    (output_dir / "refine_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Refine generated images by matching real class texture statistics.")
    parser.add_argument("--generated-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--real-dir", default="")
    parser.add_argument("--classes", nargs="*", default=None)
    parser.add_argument("--target-std-scale", type=float, default=0.75)
    parser.add_argument("--clahe-clip", type=float, default=1.5)
    parser.add_argument("--sharpen-amount", type=float, default=0.25)
    return parser.parse_args()


def main():
    args = parse_args()
    summary = refine_generated_samples(
        generated_dir=args.generated_dir,
        output_dir=args.output_dir,
        real_dir=args.real_dir or None,
        classes=args.classes,
        target_std_scale=args.target_std_scale,
        clahe_clip=args.clahe_clip,
        sharpen_amount=args.sharpen_amount,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
