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


def _save_system_workflow(output_dir):
    steps = [
        ("NEU-DET\nraw images + XML", 0.08, 0.70),
        ("ROI preprocessing\nCLAHE + denoise", 0.29, 0.70),
        ("Augmentation\nTraditional / cGAN-v2", 0.50, 0.70),
        ("Quality gate\nFID + SSIM + PSNR", 0.71, 0.70),
        ("Sample filtering\nsharpness + std", 0.91, 0.70),
        ("Classifier validation\nmulti-seed + early stop", 0.29, 0.28),
        ("Industrial readiness\nrecall + risk cost", 0.55, 0.28),
        ("Defense evidence\nfigures + reports", 0.80, 0.28),
    ]
    arrows = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)]

    fig, ax = plt.subplots(figsize=(12, 5.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    box_style = dict(boxstyle="round,pad=0.35,rounding_size=0.04", facecolor="#eef5f5", edgecolor="#2f6f73", linewidth=1.8)
    for idx, (label, x, y) in enumerate(steps):
        ax.text(x, y, label, ha="center", va="center", fontsize=10, bbox=box_style)
        if idx in {2, 6}:
            ax.text(x, y - 0.15, "key defense point", ha="center", va="center", fontsize=8, color="#8b1e1e")
    for src, dst in arrows:
        x1, y1 = steps[src][1], steps[src][2]
        x2, y2 = steps[dst][1], steps[dst][2]
        if (src, dst) == (4, 5):
            ax.annotate(
                "",
                xy=(x2 + 0.13, y2 + 0.05),
                xytext=(x1 - 0.05, y1 - 0.06),
                arrowprops=dict(arrowstyle="->", lw=1.8, color="#4b5563", connectionstyle="angle3,angleA=-90,angleB=180"),
            )
        else:
            ax.annotate(
                "",
                xy=(x2 - 0.07 if x2 > x1 else x2, y2),
                xytext=(x1 + 0.07 if x2 > x1 else x1, y1),
                arrowprops=dict(arrowstyle="->", lw=1.8, color="#4b5563", connectionstyle="arc3,rad=0.08"),
            )
    ax.set_title("DefectAugment workflow for thesis defense and industrial evaluation", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "system_workflow_industrial.png", dpi=180)
    plt.close(fig)


def _save_ratio_ablation_plot(output_dir):
    path = Path("results/ratio_ablation/cgan_v2_40ep_seed42/ratio_ablation_summary.csv")
    if not path.exists():
        return

    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            rows.append(row)
    if not rows:
        return

    rows.sort(key=lambda row: int(row["samples_per_class"]))
    samples = [int(row["samples_per_class"]) for row in rows]
    total_samples = [int(row.get("generated_total", row.get("total_generated", 0))) for row in rows]
    best_acc = [float(row["best_val_accuracy_mean"]) * 100 for row in rows]
    final_acc = [float(row["final_val_accuracy_mean"]) * 100 for row in rows]
    best_loss = [float(row["best_val_loss_mean"]) for row in rows]

    x = np.arange(len(samples))
    fig, ax1 = plt.subplots(figsize=(9.8, 4.8))
    ax1.plot(x, best_acc, marker="o", linewidth=2.2, color="#206a5d", label="Best val accuracy")
    ax1.plot(x, final_acc, marker="s", linewidth=2.0, color="#3867a6", label="Final val accuracy")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_ylim(max(80, min(final_acc + best_acc) - 4), 100.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{sample}/class\n{total} total" for sample, total in zip(samples, total_samples)])
    ax1.grid(axis="y", alpha=0.25)
    for idx, value in enumerate(best_acc):
        ax1.text(idx, value + 0.18, f"{value:.2f}%", ha="center", fontsize=8, color="#17463e")

    ax2 = ax1.twinx()
    ax2.plot(x, best_loss, marker="^", linewidth=2.0, color="#b65f2a", label="Best val loss")
    ax2.set_ylabel("Loss")
    for idx, value in enumerate(best_loss):
        ax2.text(idx + 0.04, value + 0.004, f"{value:.3f}", ha="left", fontsize=8, color="#7d3f1c")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="lower right", fontsize=8)
    ax1.set_title("cGAN-v2 40ep generated sample ratio ablation", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "ratio_ablation_cgan_v2_40ep.png", dpi=180)
    plt.close(fig)


def _save_industrial_gate_comparison(output_dir):
    experiments = [
        ("300 samples", Path("results/industrial_readiness/cgan_v2_40ep/industrial_readiness.json")),
        ("600 samples", Path("results/industrial_readiness/cgan_v2_40ep_600_seed42/industrial_readiness.json")),
    ]
    labels = []
    best_acc = []
    min_recall = []
    weighted_error = []
    passed = []
    for label, path in experiments:
        if not path.exists():
            continue
        summary = _load_summary(path)
        labels.append(label)
        best_acc.append(summary["best_val_accuracy"] * 100)
        min_recall.append(summary["min_class_recall"] * 100)
        weighted_error.append(summary["weighted_error_rate"] * 100)
        passed.append(summary["passed"])
    if len(labels) < 2:
        return

    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.2))
    metrics = [
        ("Best accuracy", best_acc, 98.0, ">=", "#206a5d"),
        ("Min class recall", min_recall, 95.0, ">=", "#3867a6"),
        ("Weighted error", weighted_error, 6.0, "<=", "#b65f2a"),
    ]
    for ax, (title, values, threshold, direction, color) in zip(axes, metrics):
        bars = ax.bar(x, values, color=color, alpha=0.88)
        ax.axhline(threshold, color="#8b1e1e", linestyle="--", linewidth=1.4, label=f"gate {direction} {threshold:.0f}%")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.22)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.4, f"{value:.2f}%", ha="center", fontsize=8)
        ax.legend(fontsize=7, loc="best")
    axes[0].set_ylabel("Percent (%)")
    fig.suptitle(
        f"Industrial readiness gate: {'PASS' if passed[-1] else 'REVIEW'} after ratio tuning",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_dir / "industrial_gate_comparison.png", dpi=180)
    plt.close(fig)


def _save_multiseed_600_plot(output_dir):
    path = Path("results/multiseed/cgan_v2_40ep_600/multiseed_runs.csv")
    if not path.exists():
        return

    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        return

    seeds = sorted({int(row["seed"]) for row in rows})
    groups = ["base", "augmented"]
    values = {
        group: [
            float(next(row["best_val_accuracy"] for row in rows if row["group"] == group and int(row["seed"]) == seed)) * 100
            for seed in seeds
        ]
        for group in groups
    }
    losses = {
        group: [
            float(next(row["best_val_loss"] for row in rows if row["group"] == group and int(row["seed"]) == seed))
            for seed in seeds
        ]
        for group in groups
    }

    x = np.arange(len(seeds))
    width = 0.34
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    axes[0].bar(x - width / 2, values["base"], width=width, color="#7a8b99", label="Base")
    axes[0].bar(x + width / 2, values["augmented"], width=width, color="#206a5d", label="cGAN-v2 600")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([str(seed) for seed in seeds])
    axes[0].set_ylabel("Best accuracy (%)")
    axes[0].set_ylim(max(88, min(values["base"] + values["augmented"]) - 4), 100.8)
    axes[0].set_title("Best validation accuracy by seed")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8, loc="lower right")
    for idx, seed in enumerate(seeds):
        axes[0].text(idx - width / 2, values["base"][idx] + 0.15, f"{values['base'][idx]:.2f}%", ha="center", fontsize=7)
        axes[0].text(idx + width / 2, values["augmented"][idx] + 0.15, f"{values['augmented'][idx]:.2f}%", ha="center", fontsize=7)

    axes[1].bar(x - width / 2, losses["base"], width=width, color="#7a8b99", label="Base")
    axes[1].bar(x + width / 2, losses["augmented"], width=width, color="#b65f2a", label="cGAN-v2 600")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([str(seed) for seed in seeds])
    axes[1].set_ylabel("Best loss")
    axes[1].set_title("Best validation loss by seed")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(fontsize=8, loc="upper right")
    for idx, seed in enumerate(seeds):
        axes[1].text(idx - width / 2, losses["base"][idx] + 0.004, f"{losses['base'][idx]:.3f}", ha="center", fontsize=7)
        axes[1].text(idx + width / 2, losses["augmented"][idx] + 0.004, f"{losses['augmented'][idx]:.3f}", ha="center", fontsize=7)

    fig.suptitle("Multi-seed validation: baseline vs cGAN-v2 600 generated samples", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_dir / "multiseed_cgan_v2_600.png", dpi=180)
    plt.close(fig)


def create_evidence_figures(output_dir="assets/figures"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_sample_grid(output_dir)
    _save_history_plot(output_dir)
    _save_result_bars(output_dir)
    _save_system_workflow(output_dir)
    _save_ratio_ablation_plot(output_dir)
    _save_industrial_gate_comparison(output_dir)
    _save_multiseed_600_plot(output_dir)
    return {"output_dir": str(output_dir), "figures": sorted(p.name for p in output_dir.glob("*.png"))}


def main():
    parser = argparse.ArgumentParser(description="Create thesis evidence figures from experiment outputs.")
    parser.add_argument("--output-dir", default="assets/figures")
    args = parser.parse_args()
    print(json.dumps(create_evidence_figures(args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
