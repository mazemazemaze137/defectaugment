import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

from src.augment.cgan_256 import export_generated_samples
from src.augment.refine_generated_samples import refine_generated_samples


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
CLASS_LABELS_ZH = {
    "crazing": "裂纹",
    "inclusion": "夹杂",
    "patches": "斑块",
    "pitted_surface": "麻点",
    "pitted-surface": "麻点",
    "rolled-in_scale": "氧化皮压入",
    "rolled-in-scale": "氧化皮压入",
    "scratches": "划痕",
}


def _setup_chinese_font():
    candidates = ["Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC"]
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False


_setup_chinese_font()


def _image_files(folder):
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted(path for path in folder.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)


def _stats(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return {
        "mean": float(np.mean(img)),
        "std": float(np.std(img)),
        "sharpness": float(cv2.Laplacian(img, cv2.CV_64F).var()),
    }


def _summarize_folder(folder):
    rows = []
    for path in _image_files(folder):
        item = _stats(path)
        if item:
            rows.append(item)
    if not rows:
        return {"count": 0, "mean": 0.0, "std": 0.0, "sharpness": 0.0}
    return {
        "count": len(rows),
        "mean": float(np.mean([row["mean"] for row in rows])),
        "std": float(np.mean([row["std"] for row in rows])),
        "sharpness": float(np.mean([row["sharpness"] for row in rows])),
    }


def _save_preview_grid(raw_dir, selected_dir, refined_dir, output_path, class_name=None, samples=6):
    raw_files = _image_files(raw_dir)
    selected_files = _image_files(selected_dir)
    refined_files = _image_files(refined_dir) if refined_dir else []
    if class_name:
        raw_files = [path for path in raw_files if path.parent.name == class_name]
        selected_files = [path for path in selected_files if path.parent.name == class_name]
        refined_files = [path for path in refined_files if path.parent.name == class_name]
    raw_files = raw_files[:samples]
    selected_files = selected_files[:samples]
    refined_files = refined_files[:samples]
    rows = [("普通随机导出", raw_files), ("质量优选导出", selected_files)]
    if refined_files:
        rows.append(("真实统计匹配", refined_files))
    if any(not files for _, files in rows):
        return None

    fig, axes = plt.subplots(len(rows), samples, figsize=(samples * 1.45, 1.55 * len(rows) + 0.4))
    if len(rows) == 1:
        axes = np.asarray([axes])
    class_label = CLASS_LABELS_ZH.get(class_name or "", class_name or "")
    for row_idx, (label, files) in enumerate(rows):
        for col_idx, path in enumerate(files):
            ax = axes[row_idx, col_idx]
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            ax.set_xticks([])
            ax.set_yticks([])
            if col_idx == 0:
                ax.set_ylabel(label, fontsize=10)
    title_suffix = f"（{class_label}）" if class_label else ""
    fig.suptitle(f"生成样本导出策略对比{title_suffix}", fontsize=13, fontweight="bold")
    fig.text(0.5, 0.02, "说明：先通过多候选质量优选减少模糊样本，再按真实类别统计匹配亮度和对比度，使生成图更接近真实工业纹理。", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def build_generation_quality_preview(
    checkpoint,
    output_dir="results/generation_quality_preview",
    real_dir="data/processed/gui_temp",
    class_names=None,
    samples_per_class=24,
    image_size=128,
    batch_size=64,
    truncation=0.85,
    oversample_factor=3,
    preview_class="pitted_surface",
):
    output_dir = Path(output_dir)
    raw_dir = output_dir / "raw_export"
    selected_dir = output_dir / "quality_selected_export"
    refined_dir = output_dir / "quality_selected_refined"
    figure_path = output_dir / "generation_quality_preview.png"

    raw_summary = export_generated_samples(
        checkpoint_path=checkpoint,
        output_dir=raw_dir,
        class_names=class_names,
        samples_per_class=samples_per_class,
        image_size=image_size,
        batch_size=batch_size,
        truncation=1.0,
        oversample_factor=1,
        quality_select=False,
    )
    selected_summary = export_generated_samples(
        checkpoint_path=checkpoint,
        output_dir=selected_dir,
        class_names=class_names,
        samples_per_class=samples_per_class,
        image_size=image_size,
        batch_size=batch_size,
        truncation=truncation,
        oversample_factor=oversample_factor,
        quality_select=True,
    )
    refine_summary = refine_generated_samples(
        generated_dir=selected_dir,
        output_dir=refined_dir,
        real_dir=real_dir,
        target_std_scale=0.90,
        clahe_clip=1.2,
        sharpen_amount=0.15,
    )
    figure = _save_preview_grid(raw_dir, selected_dir, refined_dir, figure_path, class_name=preview_class)
    summary = {
        "checkpoint": str(checkpoint),
        "output_dir": str(output_dir),
        "raw_dir": str(raw_dir),
        "selected_dir": str(selected_dir),
        "refined_dir": str(refined_dir),
        "figure": figure,
        "settings": {
            "samples_per_class": samples_per_class,
            "image_size": image_size,
            "truncation": truncation,
            "oversample_factor": oversample_factor,
            "preview_class": preview_class,
        },
        "raw_export": raw_summary,
        "quality_selected_export": selected_summary,
        "refine_summary": refine_summary,
        "raw_stats": _summarize_folder(raw_dir),
        "quality_selected_stats": _summarize_folder(selected_dir),
        "quality_selected_refined_stats": _summarize_folder(refined_dir),
    }
    (output_dir / "generation_quality_preview_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description="Compare raw cGAN export with truncation + quality-selected export.")
    parser.add_argument("--checkpoint", default="results/cgan_v2_roi_40ep/checkpoint_latest.pth")
    parser.add_argument("--output-dir", default="results/generation_quality_preview")
    parser.add_argument("--real-dir", default="data/processed/gui_temp")
    parser.add_argument("--class-names", nargs="*", default=None)
    parser.add_argument("--samples-per-class", type=int, default=24)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--truncation", type=float, default=0.85)
    parser.add_argument("--oversample-factor", type=int, default=3)
    parser.add_argument("--preview-class", default="pitted_surface")
    args = parser.parse_args()
    result = build_generation_quality_preview(
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        real_dir=args.real_dir,
        class_names=args.class_names,
        samples_per_class=args.samples_per_class,
        image_size=args.image_size,
        batch_size=args.batch_size,
        truncation=args.truncation,
        oversample_factor=args.oversample_factor,
        preview_class=args.preview_class,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
