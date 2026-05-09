import argparse
import json
from pathlib import Path

import pandas as pd

from src.evaluate.classifier_validation import parse_class_weights, run_classification_validation


DEFAULT_MODELS = ["small_cnn", "resnet18", "mobilenet_v3_small"]


def _load_or_run(run_dir, reuse_existing, **kwargs):
    summary_path = Path(run_dir) / "summary.json"
    if reuse_existing and summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    return run_classification_validation(output_dir=run_dir, **kwargs)


def _write_report(rows, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "classifier_model_comparison.csv", index=False, encoding="utf-8-sig")

    lines = [
        "# 分类器模型对比报告",
        "",
        "| 模型 | 最佳验证准确率 | 最终验证准确率 | 最佳验证损失 | 最佳轮数 | 实际轮数 | 低置信度样本数 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['model_name']} | {row['best_val_accuracy'] * 100:.2f}% | "
            f"{row['final_val_accuracy'] * 100:.2f}% | {row['best_val_loss']:.4f} | "
            f"{row['best_epoch']} | {row['completed_epochs']} | {row['low_confidence_records']} |"
        )
    best = max(rows, key=lambda row: row["best_val_accuracy"])
    lines.extend(
        [
            "",
            f"- 当前设置下最佳模型为 `{best['model_name']}`，最佳验证准确率为 {best['best_val_accuracy'] * 100:.2f}%。",
            "- 若不同模型表现接近，答辩中可强调增强数据对模型结构具有一定泛化性；若差异明显，可说明后续应继续调参或扩大模型对比。",
        ]
    )
    (output_dir / "classifier_model_comparison_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return df


def parse_args():
    parser = argparse.ArgumentParser(description="Compare downstream classifier backbones.")
    parser.add_argument("--train-dir", default="data/raw/NEU-DET/train/images")
    parser.add_argument("--val-dir", default="data/raw/NEU-DET/validation/images")
    parser.add_argument("--generated-dir", default="")
    parser.add_argument("--output-dir", default="results/classifier_model_comparison")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early-stopping-patience", type=int, default=4)
    parser.add_argument("--class-weights", default="")
    parser.add_argument("--low-confidence-threshold", type=float, default=0.7)
    parser.add_argument("--max-low-confidence-samples", type=int, default=80)
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--show-progress", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    class_weights = parse_class_weights(args.class_weights)
    rows = []
    for model_name in args.models:
        run_dir = output_dir / model_name
        summary = _load_or_run(
            run_dir,
            args.reuse_existing,
            train_dir=args.train_dir,
            val_dir=args.val_dir,
            generated_dir=args.generated_dir or None,
            image_size=args.image_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            num_workers=0,
            seed=args.seed,
            early_stopping_patience=args.early_stopping_patience,
            class_weights=class_weights,
            quiet=not args.show_progress,
            model_name=model_name,
            low_confidence_threshold=args.low_confidence_threshold,
            max_low_confidence_samples=args.max_low_confidence_samples,
        )
        rows.append(
            {
                "model_name": summary["model_name"],
                "best_val_accuracy": summary["best_val_accuracy"],
                "final_val_accuracy": summary["final_val_accuracy"],
                "best_val_loss": summary["best_val_loss"],
                "best_epoch": summary["best_epoch"],
                "completed_epochs": summary["completed_epochs"],
                "elapsed_seconds": summary["elapsed_seconds"],
                "low_confidence_records": summary.get("low_confidence", {}).get("num_records", 0),
                "output_dir": summary["output_dir"],
            }
        )
    df = _write_report(rows, output_dir)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
