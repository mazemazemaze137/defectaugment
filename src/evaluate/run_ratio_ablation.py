import argparse
import json
from pathlib import Path

import pandas as pd

from src.augment.filter_generated_samples import filter_generated_samples
from src.evaluate.classifier_validation import parse_class_weights, run_classification_validation


def _parse_int_list(values):
    if isinstance(values, str):
        values = [values]
    parsed = []
    for value in values:
        for item in str(value).split(","):
            item = item.strip()
            if item:
                parsed.append(int(item))
    return parsed


def _load_or_run(run_dir, reuse_existing, **kwargs):
    summary_path = Path(run_dir) / "summary.json"
    if reuse_existing and summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    return run_classification_validation(output_dir=run_dir, **kwargs)


def _write_report(df, output_dir):
    output_dir = Path(output_dir)
    df.to_csv(output_dir / "ratio_ablation_runs.csv", index=False, encoding="utf-8-sig")
    grouped = (
        df.groupby("samples_per_class")
        .agg(
            seeds=("seed", "count"),
            generated_total=("generated_total", "max"),
            best_val_accuracy_mean=("best_val_accuracy", "mean"),
            best_val_accuracy_std=("best_val_accuracy", "std"),
            final_val_accuracy_mean=("final_val_accuracy", "mean"),
            best_val_loss_mean=("best_val_loss", "mean"),
        )
        .reset_index()
        .sort_values("samples_per_class")
    )
    grouped.to_csv(output_dir / "ratio_ablation_summary.csv", index=False, encoding="utf-8-sig")

    lines = [
        "# 生成样本比例消融实验报告",
        "",
        "| 每类生成样本 | 生成样本总数 | 种子数 | 最佳准确率均值 | 最佳准确率标准差 | 最终准确率均值 | 最佳损失均值 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in grouped.iterrows():
        std = 0.0 if pd.isna(row["best_val_accuracy_std"]) else row["best_val_accuracy_std"]
        lines.append(
            f"| {int(row['samples_per_class'])} | {int(row['generated_total'])} | {int(row['seeds'])} | "
            f"{row['best_val_accuracy_mean'] * 100:.2f}% | {std * 100:.2f}% | "
            f"{row['final_val_accuracy_mean'] * 100:.2f}% | {row['best_val_loss_mean']:.4f} |"
        )
    lines.extend(
        [
            "",
            "- 该实验用于回答“生成样本是否越多越好”。",
            "- 若中等比例优于大比例，论文中应强调质量筛选和类别均衡比单纯扩充数量更重要。",
            "- 若大比例继续提升，应结合工业就绪度报告检查最低类别召回率是否同步改善。",
        ]
    )
    (output_dir / "ratio_ablation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return grouped


def parse_args():
    parser = argparse.ArgumentParser(description="Run generated sample ratio ablation for classifier validation.")
    parser.add_argument("--train-dir", default="data/raw/NEU-DET/train/images")
    parser.add_argument("--val-dir", default="data/raw/NEU-DET/validation/images")
    parser.add_argument("--generated-source-dir", required=True)
    parser.add_argument("--output-dir", default="results/ratio_ablation")
    parser.add_argument("--samples-per-class", nargs="+", default=["25", "50", "100"])
    parser.add_argument("--seeds", nargs="+", default=["42"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--early-stopping-patience", type=int, default=4)
    parser.add_argument("--min-sharpness", type=float, default=5.0)
    parser.add_argument("--min-std", type=float, default=8.0)
    parser.add_argument("--class-weights", default="")
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--show-progress", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    subsets_dir = output_dir / "subsets"
    runs_dir = output_dir / "runs"
    output_dir.mkdir(parents=True, exist_ok=True)
    class_weights = parse_class_weights(args.class_weights)
    rows = []

    for samples_per_class in _parse_int_list(args.samples_per_class):
        subset_dir = subsets_dir / f"gan_{samples_per_class}_per_class"
        if not (args.reuse_existing and (subset_dir / "filter_summary.json").exists()):
            filter_generated_samples(
                input_dir=args.generated_source_dir,
                output_dir=subset_dir,
                max_per_class=samples_per_class,
                min_sharpness=args.min_sharpness,
                min_std=args.min_std,
                adaptive_min_per_class=samples_per_class,
            )
        filter_summary = json.loads((subset_dir / "filter_summary.json").read_text(encoding="utf-8"))
        generated_total = int(filter_summary["total_selected"])

        for seed in _parse_int_list(args.seeds):
            run_dir = runs_dir / f"gan_{samples_per_class}_seed{seed}"
            summary = _load_or_run(
                run_dir,
                args.reuse_existing,
                train_dir=args.train_dir,
                val_dir=args.val_dir,
                generated_dir=subset_dir,
                image_size=args.image_size,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                num_workers=0,
                seed=seed,
                early_stopping_patience=args.early_stopping_patience,
                class_weights=class_weights,
                quiet=not args.show_progress,
            )
            rows.append(
                {
                    "samples_per_class": samples_per_class,
                    "generated_total": generated_total,
                    "seed": seed,
                    "best_val_accuracy": summary["best_val_accuracy"],
                    "final_val_accuracy": summary["final_val_accuracy"],
                    "best_val_loss": summary["best_val_loss"],
                    "best_epoch": summary.get("best_epoch"),
                    "completed_epochs": summary.get("completed_epochs"),
                    "output_dir": summary["output_dir"],
                }
            )

    grouped = _write_report(pd.DataFrame(rows), output_dir)
    print(grouped.to_string(index=False))


if __name__ == "__main__":
    main()
