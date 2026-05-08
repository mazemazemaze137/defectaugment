import argparse
import json
from pathlib import Path

import pandas as pd

from src.evaluate.classifier_validation import parse_class_weights, run_classification_validation
from src.evaluate.industrial_readiness import analyze_industrial_readiness


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


def _summarize(rows, output_dir):
    df = pd.DataFrame(rows)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "multiseed_runs.csv", index=False, encoding="utf-8-sig")

    grouped = (
        df.groupby("group")
        .agg(
            seeds=("seed", "count"),
            best_val_accuracy_mean=("best_val_accuracy", "mean"),
            best_val_accuracy_std=("best_val_accuracy", "std"),
            final_val_accuracy_mean=("final_val_accuracy", "mean"),
            final_val_accuracy_std=("final_val_accuracy", "std"),
            best_val_loss_mean=("best_val_loss", "mean"),
            best_val_loss_std=("best_val_loss", "std"),
        )
        .reset_index()
    )
    grouped.to_csv(output_dir / "multiseed_summary.csv", index=False, encoding="utf-8-sig")

    base = grouped[grouped["group"] == "base"]
    augmented = grouped[grouped["group"] == "augmented"]
    report = ["# 多随机种子验证报告", ""]
    report.append("| 实验组 | 种子数 | 最佳准确率均值 | 最佳准确率标准差 | 最终准确率均值 | 最佳损失均值 |")
    report.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for _, row in grouped.iterrows():
        report.append(
            f"| {row['group']} | {int(row['seeds'])} | {row['best_val_accuracy_mean'] * 100:.2f}% | "
            f"{(0 if pd.isna(row['best_val_accuracy_std']) else row['best_val_accuracy_std']) * 100:.2f}% | "
            f"{row['final_val_accuracy_mean'] * 100:.2f}% | {row['best_val_loss_mean']:.4f} |"
        )
    if not base.empty and not augmented.empty:
        delta = augmented.iloc[0]["best_val_accuracy_mean"] - base.iloc[0]["best_val_accuracy_mean"]
        report.extend(
            [
                "",
                f"- 增强组平均最佳验证准确率相对基线变化：{delta * 100:+.2f} 个百分点。",
                "- 若增强组最终准确率波动较大，论文中应强调早停和最佳模型保存的重要性。",
            ]
        )
    (output_dir / "multiseed_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return grouped


def parse_args():
    parser = argparse.ArgumentParser(description="Run baseline and augmented classifier validation across multiple seeds.")
    parser.add_argument("--train-dir", default="data/raw/NEU-DET/train/images")
    parser.add_argument("--val-dir", default="data/raw/NEU-DET/validation/images")
    parser.add_argument("--generated-dir", required=True)
    parser.add_argument("--output-dir", default="results/multiseed_auto")
    parser.add_argument("--seeds", nargs="+", default=["42", "7", "123"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--early-stopping-patience", type=int, default=4)
    parser.add_argument("--class-weights", default="")
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--industrial-report", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    class_weights = parse_class_weights(args.class_weights)
    rows = []

    for seed in _parse_int_list(args.seeds):
        common = {
            "train_dir": args.train_dir,
            "val_dir": args.val_dir,
            "image_size": args.image_size,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "num_workers": 0,
            "seed": seed,
            "early_stopping_patience": args.early_stopping_patience,
            "class_weights": class_weights,
        }
        base_dir = output_dir / f"base_seed{seed}"
        aug_dir = output_dir / f"augmented_seed{seed}"
        base = _load_or_run(base_dir, args.reuse_existing, generated_dir=None, **common)
        augmented = _load_or_run(aug_dir, args.reuse_existing, generated_dir=args.generated_dir, **common)
        for group, summary in [("base", base), ("augmented", augmented)]:
            rows.append(
                {
                    "group": group,
                    "seed": seed,
                    "best_val_accuracy": summary["best_val_accuracy"],
                    "final_val_accuracy": summary["final_val_accuracy"],
                    "best_val_loss": summary["best_val_loss"],
                    "best_epoch": summary.get("best_epoch"),
                    "completed_epochs": summary.get("completed_epochs"),
                    "elapsed_seconds": summary["elapsed_seconds"],
                    "output_dir": summary["output_dir"],
                }
            )
        if args.industrial_report:
            analyze_industrial_readiness(aug_dir, output_dir=aug_dir / "industrial_readiness")

    grouped = _summarize(rows, output_dir)
    print(grouped.to_string(index=False))


if __name__ == "__main__":
    main()
