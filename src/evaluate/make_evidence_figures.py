import argparse
import csv
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


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
        ("真实ROI样本", Path("data/processed/gui_temp/pitted_surface")),
        ("20轮原始生成", Path("results/cgan_v2_roi_20ep/export_100_per_class/pitted_surface")),
        ("20轮后处理筛选", Path("results/cgan_v2_roi_20ep/export_refined_pitted_filtered_50/pitted_surface")),
        ("40轮筛选结果", Path("results/cgan_v2_roi_40ep/export_filtered_50_per_class/pitted_surface")),
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
    fig.suptitle("麻点缺陷样本对比：真实ROI、早期生成、后处理筛选与40轮结果", fontsize=14, fontweight="bold")
    fig.text(
        0.5,
        0.018,
        "说明：每行展示同一类别的5个样本，用于观察纹理清晰度、灰度对比度和生成样本多样性。",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(output_dir / "pitted_surface_refinement_grid.png", dpi=180, bbox_inches="tight")
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
        ("原始基线", "results/ablation_earlystop/base/history.csv"),
        ("传统增强600张", "results/ablation_earlystop/traditional_600/history.csv"),
        ("cGAN-v2 20轮自适应筛选", "results/ablation_earlystop/cgan_v2_adaptive_300/history.csv"),
        ("cGAN-v2 40轮筛选", "results/ablation_earlystop/cgan_v2_40ep_filtered_300/history.csv"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for label, path in experiments:
        if not Path(path).exists():
            continue
        history = _read_history(path)
        axes[0].plot(history["epoch"], history["val_accuracy"], marker="o", linewidth=1.8, label=label)
        axes[1].plot(history["epoch"], history["val_loss"], marker="o", linewidth=1.8, label=label)

    axes[0].set_title("验证准确率变化")
    axes[0].set_xlabel("训练轮数")
    axes[0].set_ylabel("准确率（%）")
    axes[0].grid(alpha=0.25)
    axes[1].set_title("验证损失变化")
    axes[1].set_xlabel("训练轮数")
    axes[1].set_ylabel("损失值")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "classifier_history_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _load_summary(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _save_result_bars(output_dir):
    experiments = [
        ("原始基线", "results/ablation_earlystop/base/summary.json"),
        ("cGAN-v2 20轮", "results/ablation_earlystop/cgan_v2_adaptive_300/summary.json"),
        ("cGAN-v2 40轮", "results/ablation_earlystop/cgan_v2_40ep_filtered_300/summary.json"),
        ("传统增强", "results/ablation_earlystop/traditional_600/summary.json"),
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
    bars = ax1.bar(x - 0.18, best_acc, width=0.36, color="#2f6f73", label="最佳验证准确率")
    ax1.set_ylabel("最佳准确率（%）")
    ax1.set_ylim(max(80, min(best_acc) - 5), 100)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, best_acc):
        ax1.text(bar.get_x() + bar.get_width() / 2, value + 0.15, f"{value:.2f}%", ha="center", fontsize=8)

    ax2 = ax1.twinx()
    ax2.plot(x + 0.18, best_loss, color="#b65f2a", marker="o", linewidth=2, label="最佳验证损失")
    ax2.set_ylabel("最佳损失")
    for idx, value in enumerate(best_loss):
        ax2.text(idx + 0.18, value + 0.006, f"{value:.3f}", ha="center", fontsize=8, color="#7d3f1c")

    lines, line_labels = ax1.get_legend_handles_labels()
    lines2, line_labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, line_labels + line_labels2, loc="lower right", fontsize=8)
    ax1.set_title("不同增强方案的下游分类结果对比", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "experiment_result_bars.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_system_workflow(output_dir):
    steps = [
        ("NEU-DET数据\n原图+XML标注", 0.18, 0.78),
        ("ROI预处理\nCLAHE+去噪", 0.50, 0.78),
        ("数据增强\n传统增强/cGAN-v2", 0.82, 0.78),
        ("质量评估\nFID+SSIM+PSNR", 0.82, 0.52),
        ("样本筛选\n清晰度+灰度方差", 0.50, 0.52),
        ("分类验证\n多随机种子+早停", 0.18, 0.52),
        ("工业门槛\n召回率+风险代价", 0.18, 0.26),
        ("答辩证据\n图表+报告+烟测", 0.50, 0.26),
    ]
    arrows = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)]

    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    box_style = dict(boxstyle="round,pad=0.35,rounding_size=0.04", facecolor="#eef5f5", edgecolor="#2f6f73", linewidth=1.8)
    for idx, (label, x, y) in enumerate(steps):
        ax.text(x, y, label, ha="center", va="center", fontsize=12, bbox=box_style)
        if idx in {2, 6}:
            ax.text(x, y - 0.12, "答辩重点", ha="center", va="center", fontsize=9, color="#8b1e1e")
    for src, dst in arrows:
        x1, y1 = steps[src][1], steps[src][2]
        x2, y2 = steps[dst][1], steps[dst][2]
        if abs(y2 - y1) < 0.02:
            start = (x1 + (0.12 if x2 > x1 else -0.12), y1)
            end = (x2 - (0.12 if x2 > x1 else -0.12), y2)
        else:
            start = (x1, y1 - 0.08)
            end = (x2, y2 + 0.08)
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops=dict(arrowstyle="->", lw=1.9, color="#4b5563", connectionstyle="arc3,rad=0.08"),
        )
    ax.set_title("工业缺陷数据增强系统闭环流程", fontsize=16, fontweight="bold")
    ax.text(0.5, 0.05, "说明：从真实缺陷数据出发，经过生成、筛选和下游验证，最终判断是否具备试运行价值。", ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(output_dir / "system_workflow_industrial.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_ratio_ablation_plot(output_dir):
    path = Path("results/ratio_ablation/cgan_v2_40ep_multiseed/ratio_ablation_summary.csv")
    if not path.exists():
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
    best_acc_std = [float(row.get("best_val_accuracy_std") or 0) * 100 for row in rows]
    final_acc = [float(row["final_val_accuracy_mean"]) * 100 for row in rows]
    best_loss = [float(row["best_val_loss_mean"]) for row in rows]

    x = np.arange(len(samples))
    fig, ax1 = plt.subplots(figsize=(9.8, 4.8))
    ax1.plot(x, best_acc, marker="o", linewidth=2.2, color="#206a5d", label="最佳验证准确率")
    if any(value > 0 for value in best_acc_std):
        ax1.fill_between(
            x,
            np.array(best_acc) - np.array(best_acc_std),
            np.array(best_acc) + np.array(best_acc_std),
            color="#206a5d",
            alpha=0.14,
            label="最佳准确率标准差",
        )
    ax1.plot(x, final_acc, marker="s", linewidth=2.0, color="#3867a6", label="最终验证准确率")
    ax1.set_ylabel("准确率（%）")
    ax1.set_ylim(max(80, min(final_acc + best_acc) - 6), 100.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"每类{sample}张\n共{total}张" for sample, total in zip(samples, total_samples)])
    ax1.grid(axis="y", alpha=0.25)
    for idx, value in enumerate(best_acc):
        ax1.text(idx, value + 0.18, f"{value:.2f}%", ha="center", fontsize=8, color="#17463e")

    ax2 = ax1.twinx()
    ax2.plot(x, best_loss, marker="^", linewidth=2.0, color="#b65f2a", label="最佳验证损失")
    ax2.set_ylabel("损失值")
    for idx, value in enumerate(best_loss):
        ax2.text(idx + 0.04, value + 0.004, f"{value:.3f}", ha="left", fontsize=8, color="#7d3f1c")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="lower right", fontsize=8)
    ax1.set_title("cGAN-v2 40轮生成样本比例消融（3随机种子）", fontsize=13, fontweight="bold")
    ax1.text(
        0.55,
        ax1.get_ylim()[0] + 0.7,
        "结论：每类50张平均最佳准确率最高；每类100张最终准确率更稳",
        fontsize=9,
        color="#8b1e1e",
    )
    fig.tight_layout()
    fig.savefig(output_dir / "ratio_ablation_cgan_v2_40ep.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_industrial_gate_comparison(output_dir):
    experiments = [
        ("300张方案", Path("results/industrial_readiness/cgan_v2_40ep/industrial_readiness.json")),
        ("600张方案", Path("results/industrial_readiness/cgan_v2_40ep_600_seed42/industrial_readiness.json")),
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
        ("最佳准确率", best_acc, 98.0, "≥", "#206a5d"),
        ("最低类别召回率", min_recall, 95.0, "≥", "#3867a6"),
        ("代价加权错误率", weighted_error, 6.0, "≤", "#b65f2a"),
    ]
    for ax, (title, values, threshold, direction, color) in zip(axes, metrics):
        bars = ax.bar(x, values, color=color, alpha=0.88)
        ax.axhline(threshold, color="#8b1e1e", linestyle="--", linewidth=1.4, label=f"门槛 {direction} {threshold:.0f}%")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.22)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.4, f"{value:.2f}%", ha="center", fontsize=8)
        ax.legend(fontsize=7, loc="best")
    axes[0].set_ylabel("百分比（%）")
    fig.suptitle(
        f"工业应用门槛对比：比例调节后600张方案{'通过' if passed[-1] else '需复核'}",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_dir / "industrial_gate_comparison.png", dpi=180, bbox_inches="tight")
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
    axes[0].bar(x - width / 2, values["base"], width=width, color="#7a8b99", label="原始基线")
    axes[0].bar(x + width / 2, values["augmented"], width=width, color="#206a5d", label="cGAN-v2增强600张")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([str(seed) for seed in seeds])
    axes[0].set_ylabel("最佳准确率（%）")
    axes[0].set_ylim(max(88, min(values["base"] + values["augmented"]) - 4), 100.8)
    axes[0].set_title("不同随机种子下的最佳准确率")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8, loc="lower right")
    for idx, seed in enumerate(seeds):
        axes[0].text(idx - width / 2, values["base"][idx] + 0.15, f"{values['base'][idx]:.2f}%", ha="center", fontsize=7)
        axes[0].text(idx + width / 2, values["augmented"][idx] + 0.15, f"{values['augmented'][idx]:.2f}%", ha="center", fontsize=7)

    axes[1].bar(x - width / 2, losses["base"], width=width, color="#7a8b99", label="原始基线")
    axes[1].bar(x + width / 2, losses["augmented"], width=width, color="#b65f2a", label="cGAN-v2增强600张")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([str(seed) for seed in seeds])
    axes[1].set_ylabel("最佳损失")
    axes[1].set_title("不同随机种子下的最佳损失")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(fontsize=8, loc="upper right")
    for idx, seed in enumerate(seeds):
        axes[1].text(idx - width / 2, losses["base"][idx] + 0.004, f"{losses['base'][idx]:.3f}", ha="center", fontsize=7)
        axes[1].text(idx + width / 2, losses["augmented"][idx] + 0.004, f"{losses['augmented'][idx]:.3f}", ha="center", fontsize=7)

    fig.suptitle("多随机种子复验：原始基线与cGAN-v2 600张增强方案对比", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_dir / "multiseed_cgan_v2_600.png", dpi=180, bbox_inches="tight")
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
