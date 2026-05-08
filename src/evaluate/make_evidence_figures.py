import argparse
import csv
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _image_files(folder):
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted(
        path for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _read_gray(path, size=128):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


def _save_sample_grid(output_dir):
    groups = [
        ("Real ROI", Path("data/processed/gui_temp/pitted_surface")),
        ("20ep raw cGAN-v2", Path("results/cgan_v2_roi_20ep/export_100_per_class/pitted_surface")),
        ("20ep refined", Path("results/cgan_v2_roi_20ep/export_refined_pitted_filtered_50/pitted_surface")),
        ("40ep filtered", Path("results/cgan_v2_roi_40ep/export_filtered_50_per_class/pitted_surface")),
    ]

    rows = []
    for title, folder in groups:
        files = _image_files(folder)[:5]
        if len(files) < 5:
            raise FileNotFoundError(f"Need at least 5 images in {folder}")
        rows.append((title, files))

    fig, axes = plt.subplots(len(rows), 5, figsize=(10, 7.2))
    for row_idx, (title, files) in enumerate(rows):
        for col_idx, path in enumerate(files):
            ax = axes[row_idx, col_idx]
            ax.imshow(_read_gray(path), cmap="gray", vmin=0, vmax=255)
            ax.set_xticks([])
            ax.set_yticks([])
            if col_idx == 0:
                ax.set_ylabel(title, fontsize=10)
    fig.suptitle("pitted_surface sample comparison", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_dir / "pitted_surface_refinement_grid.png", dpi=180)
    plt.close(fig)


def _read_history(path):
    with Path(path).open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    return {
        "epoch": [int(row["epoch"]) for row in rows],
        "val_accuracy": [float(row["val_accuracy"]) * 100 for row in rows],
        "val_loss": [float(row["val_loss"]) for row in rows],
    }


def _save_history_plot(output_dir):
    experiments = [
        ("Base", "results/ablation_earlystop/base/history.csv"),
        ("Traditional 600", "results/ablation_earlystop/traditional_600/history.csv"),
        ("cGAN-v2 20ep adaptive", "results/ablation_earlystop/cgan_v2_adaptive_300/history.csv"),
        ("cGAN-v2 40ep filtered", "results/ablation_earlystop/cgan_v2_40ep_filtered_300/history.csv"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for label, path in experiments:
        if not Path(path).exists():
            continue
        history = _read_history(path)
        axes[0].plot(history["epoch"], history["val_accuracy"], marker="o", linewidth=1.8, label=label)
        axes[1].plot(history["epoch"], history["val_loss"], marker="o", linewidth=1.8, label=label)

    axes[0].set_title("Validation accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].grid(alpha=0.25)
    axes[1].set_title("Validation loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "classifier_history_comparison.png", dpi=180)
    plt.close(fig)


def _load_summary(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _save_result_bars(output_dir):
    experiments = [
        ("Base", "results/ablation_earlystop/base/summary.json"),
        ("cGAN-v2 20ep", "results/ablation_earlystop/cgan_v2_adaptive_300/summary.json"),
        ("cGAN-v2 40ep", "results/ablation_earlystop/cgan_v2_40ep_filtered_300/summary.json"),
        ("Traditional", "results/ablation_earlystop/traditional_600/summary.json"),
    ]

    labels = []
    best_acc = []
    best_loss = []
    for label, path in experiments:
        if not Path(path).exists():
            continue
        summary = _load_summary(path)
        labels.append(label)
        best_acc.append(summary["best_val_accuracy"] * 100)
        best_loss.append(summary["best_val_loss"])

    x = np.arange(len(labels))
    fig, ax1 = plt.subplots(figsize=(9.5, 4.5))
    bars = ax1.bar(x - 0.18, best_acc, width=0.36, color="#2f6f73", label="Best val acc")
    ax1.set_ylabel("Best accuracy (%)")
    ax1.set_ylim(max(80, min(best_acc) - 5), 100)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, best_acc):
        ax1.text(bar.get_x() + bar.get_width() / 2, value + 0.15, f"{value:.2f}%", ha="center", fontsize=8)

    ax2 = ax1.twinx()
    ax2.plot(x + 0.18, best_loss, color="#b65f2a", marker="o", linewidth=2, label="Best val loss")
    ax2.set_ylabel("Best loss")
    for idx, value in enumerate(best_loss):
        ax2.text(idx + 0.18, value + 0.006, f"{value:.3f}", ha="center", fontsize=8, color="#7d3f1c")

    lines, line_labels = ax1.get_legend_handles_labels()
    lines2, line_labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, line_labels + line_labels2, loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "experiment_result_bars.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Create thesis evidence figures from experiment outputs.")
    parser.add_argument("--output-dir", default="assets/figures")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_sample_grid(output_dir)
    _save_history_plot(output_dir)
    _save_result_bars(output_dir)
    print(json.dumps({"output_dir": str(output_dir), "figures": sorted(p.name for p in output_dir.glob("*.png"))}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
