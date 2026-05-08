import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_SEVERITY_WEIGHTS = {
    "crazing": 5.0,
    "inclusion": 4.0,
    "patches": 3.0,
    "pitted-surface": 4.0,
    "pitted_surface": 4.0,
    "rolled-in-scale": 3.0,
    "rolled-in_scale": 3.0,
    "scratches": 4.0,
}


def _normalize_class_name(name):
    return str(name).strip().lower().replace("_", "-")


def _load_summary(result_dir):
    summary_path = Path(result_dir) / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"summary.json not found: {summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _load_confusion(result_dir):
    confusion_path = Path(result_dir) / "confusion_matrix.csv"
    if not confusion_path.exists():
        raise FileNotFoundError(f"confusion_matrix.csv not found: {confusion_path}")
    return np.loadtxt(confusion_path, delimiter=",", dtype=np.int64)


def _severity_for(class_name, overrides):
    normalized = _normalize_class_name(class_name)
    if normalized in overrides:
        return float(overrides[normalized])
    return float(DEFAULT_SEVERITY_WEIGHTS.get(normalized, 3.0))


def parse_severity_overrides(text):
    overrides = {}
    if not text:
        return overrides

    for item in text.split(","):
        if not item.strip():
            continue
        if "=" not in item:
            raise ValueError(f"Invalid severity item: {item}. Expected class=value")
        name, value = item.split("=", 1)
        overrides[_normalize_class_name(name)] = float(value)
    return overrides


def analyze_industrial_readiness(
    result_dir,
    output_dir=None,
    min_best_accuracy=0.98,
    min_class_recall=0.95,
    max_weighted_error=0.06,
    severity_overrides=None,
):
    result_dir = Path(result_dir)
    output_dir = Path(output_dir) if output_dir else result_dir / "industrial_readiness"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = _load_summary(result_dir)
    confusion = _load_confusion(result_dir)
    class_names = summary["class_names"]
    severity_overrides = severity_overrides or {}

    if confusion.shape[0] != len(class_names) or confusion.shape[1] != len(class_names):
        raise ValueError("Confusion matrix shape does not match class_names in summary.json")

    rows = []
    total_weighted_errors = 0.0
    total_weighted_samples = 0.0
    total_errors = 0
    total_samples = int(confusion.sum())

    for idx, class_name in enumerate(class_names):
        tp = int(confusion[idx, idx])
        support = int(confusion[idx, :].sum())
        predicted = int(confusion[:, idx].sum())
        missed = int(support - tp)
        false_alarm = int(predicted - tp)
        recall = tp / support if support else 0.0
        precision = tp / predicted if predicted else 0.0
        error_rate = missed / support if support else 0.0
        severity = _severity_for(class_name, severity_overrides)
        weighted_error = missed * severity
        weighted_support = support * severity
        total_weighted_errors += weighted_error
        total_weighted_samples += weighted_support
        total_errors += missed

        rows.append(
            {
                "class_name": class_name,
                "severity_weight": severity,
                "support": support,
                "correct": tp,
                "wrong": missed,
                "recall": recall,
                "precision": precision,
                "class_error_rate": error_rate,
                "weighted_error": weighted_error,
            }
        )

    weighted_error_rate = total_weighted_errors / total_weighted_samples if total_weighted_samples else 0.0
    overall_error_rate = total_errors / total_samples if total_samples else 0.0
    min_recall = min((row["recall"] for row in rows), default=0.0)
    best_accuracy = float(summary["best_val_accuracy"])
    final_accuracy = float(summary["final_val_accuracy"])

    gates = {
        "best_accuracy": {
            "value": best_accuracy,
            "threshold": min_best_accuracy,
            "passed": best_accuracy >= min_best_accuracy,
        },
        "min_class_recall": {
            "value": min_recall,
            "threshold": min_class_recall,
            "passed": min_recall >= min_class_recall,
        },
        "weighted_error_rate": {
            "value": weighted_error_rate,
            "threshold": max_weighted_error,
            "passed": weighted_error_rate <= max_weighted_error,
        },
    }
    passed = all(item["passed"] for item in gates.values())

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "industrial_class_metrics.csv", index=False, encoding="utf-8-sig")
    _save_recall_plot(df, output_dir / "industrial_recall_by_class.png", min_class_recall)

    result = {
        "result_dir": str(result_dir),
        "output_dir": str(output_dir),
        "device": summary.get("device", ""),
        "best_epoch": summary.get("best_epoch"),
        "completed_epochs": summary.get("completed_epochs", summary.get("epochs")),
        "best_val_accuracy": best_accuracy,
        "final_val_accuracy": final_accuracy,
        "overall_error_rate": overall_error_rate,
        "weighted_error_rate": weighted_error_rate,
        "min_class_recall": min_recall,
        "total_validation_samples": total_samples,
        "total_errors": total_errors,
        "gates": gates,
        "passed": passed,
        "class_metrics": rows,
        "recommendations": _build_recommendations(rows, gates, passed),
    }

    (output_dir / "industrial_readiness.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "industrial_readiness_report.md").write_text(
        build_markdown_report(result),
        encoding="utf-8",
    )
    return result


def _save_recall_plot(df, output_path, min_class_recall):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = ["#2f6f73" if value >= min_class_recall else "#b65f2a" for value in df["recall"]]
    bars = ax.bar(df["class_name"], df["recall"] * 100, color=colors)
    ax.axhline(min_class_recall * 100, color="#8b1e1e", linestyle="--", linewidth=1.5, label="Recall gate")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Recall (%)")
    ax.set_title("Industrial recall by defect class")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, df["recall"] * 100):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1.0, f"{value:.1f}%", ha="center", fontsize=8)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _build_recommendations(rows, gates, passed):
    recommendations = []
    weak_classes = [row for row in rows if row["recall"] < gates["min_class_recall"]["threshold"]]
    costly_classes = sorted(rows, key=lambda row: row["weighted_error"], reverse=True)[:2]

    if passed:
        recommendations.append("当前结果达到设定上线门槛，可进入小规模试运行或人工复核并行验证。")
    else:
        recommendations.append("当前结果未完全达到设定上线门槛，建议作为离线辅助工具或继续优化后再试运行。")

    if not gates["best_accuracy"]["passed"]:
        recommendations.append("整体最佳准确率不足，建议增加真实样本、传统增强和多随机种子训练。")
    if not gates["min_class_recall"]["passed"]:
        names = "、".join(row["class_name"] for row in weak_classes)
        recommendations.append(f"以下类别召回率低于门槛：{names}。建议优先补充这些类别的真实样本和高质量生成样本。")
    if not gates["weighted_error_rate"]["passed"]:
        names = "、".join(row["class_name"] for row in costly_classes if row["weighted_error"] > 0)
        recommendations.append(f"代价加权错误率偏高，主要风险类别为：{names}。建议提高这些类别的损失权重或筛选阈值。")

    recommendations.append("工业部署时建议保留人工复核通道，并记录模型低置信度样本用于后续主动学习。")
    return recommendations


def build_markdown_report(result):
    status = "通过" if result["passed"] else "未通过"
    lines = [
        "# 工业应用就绪度评估报告",
        "",
        "## 总体结论",
        "",
        f"- 评估状态：{status}",
        f"- 最佳验证准确率：{result['best_val_accuracy'] * 100:.2f}%",
        f"- 最终验证准确率：{result['final_val_accuracy'] * 100:.2f}%",
        f"- 最低类别召回率：{result['min_class_recall'] * 100:.2f}%",
        f"- 代价加权错误率：{result['weighted_error_rate'] * 100:.2f}%",
        f"- 验证样本数：{result['total_validation_samples']}",
        "",
        "## 上线门槛",
        "",
        "| 指标 | 当前值 | 门槛 | 结果 |",
        "| --- | ---: | ---: | --- |",
    ]
    gate_labels = {
        "best_accuracy": "最佳验证准确率",
        "min_class_recall": "最低类别召回率",
        "weighted_error_rate": "代价加权错误率",
    }
    for key, item in result["gates"].items():
        op = ">=" if key != "weighted_error_rate" else "<="
        lines.append(
            f"| {gate_labels[key]} | {item['value'] * 100:.2f}% | {op} {item['threshold'] * 100:.2f}% | "
            f"{'通过' if item['passed'] else '未通过'} |"
        )

    lines.extend(
        [
            "",
            "## 类别级风险",
            "",
            "| 类别 | 严重度权重 | 验证样本 | 正确 | 错误 | 召回率 | 精确率 | 加权错误 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in result["class_metrics"]:
        lines.append(
            f"| {row['class_name']} | {row['severity_weight']:.1f} | {row['support']} | "
            f"{row['correct']} | {row['wrong']} | {row['recall'] * 100:.2f}% | "
            f"{row['precision'] * 100:.2f}% | {row['weighted_error']:.1f} |"
        )

    lines.extend(["", "## 建议", ""])
    lines.extend(f"- {item}" for item in result["recommendations"])
    lines.extend(
        [
            "",
            "## 输出文件",
            "",
            f"- 类别指标：`{Path(result['output_dir']) / 'industrial_class_metrics.csv'}`",
            f"- 召回率图：`{Path(result['output_dir']) / 'industrial_recall_by_class.png'}`",
            f"- JSON 结果：`{Path(result['output_dir']) / 'industrial_readiness.json'}`",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(description="Build an industrial readiness report from classifier validation results.")
    parser.add_argument("--result-dir", required=True, help="Classifier validation result directory")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--min-best-accuracy", type=float, default=0.98)
    parser.add_argument("--min-class-recall", type=float, default=0.95)
    parser.add_argument("--max-weighted-error", type=float, default=0.06)
    parser.add_argument(
        "--severity",
        default="",
        help="Optional comma-separated class severity overrides, e.g. crazing=5,inclusion=4",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    result = analyze_industrial_readiness(
        result_dir=args.result_dir,
        output_dir=args.output_dir,
        min_best_accuracy=args.min_best_accuracy,
        min_class_recall=args.min_class_recall,
        max_weighted_error=args.max_weighted_error,
        severity_overrides=parse_severity_overrides(args.severity),
    )
    print(json.dumps({k: result[k] for k in ["passed", "best_val_accuracy", "min_class_recall", "weighted_error_rate", "output_dir"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
