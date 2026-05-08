import argparse
import json
from pathlib import Path


def _load_summary(run_dir):
    summary_path = Path(run_dir) / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"summary.json not found: {summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _percent(value):
    return f"{value * 100:.2f}%"


def build_classifier_report(base_dir, augmented_dir):
    base = _load_summary(base_dir)
    augmented = _load_summary(augmented_dir)

    final_delta = augmented["final_val_accuracy"] - base["final_val_accuracy"]
    best_delta = augmented["best_val_accuracy"] - base["best_val_accuracy"]
    extra_samples = augmented["counts"]["train_generated"]
    total_delta = augmented["counts"]["train_total"] - base["counts"]["train_total"]
    time_delta = augmented["elapsed_seconds"] - base["elapsed_seconds"]

    lines = [
        "# 下游分类验证实验报告",
        "",
        "## 实验设置",
        "",
        f"- 训练设备：{base['device']}",
        f"- 类别数量：{len(base['class_names'])}",
        f"- 缺陷类别：{', '.join(base['class_names'])}",
        f"- 训练轮数：{base['epochs']}",
        f"- Batch Size：{base['batch_size']}",
        f"- 输入尺寸：{base['image_size']}x{base['image_size']}",
        f"- 学习率：{base['lr']}",
        "",
        "## 样本规模",
        "",
        "| 实验组 | 原始训练样本 | 生成样本 | 训练总数 | 验证样本 |",
        "| --- | ---: | ---: | ---: | ---: |",
        (
            f"| 原始数据基线 | {base['counts']['train_original']} | "
            f"{base['counts']['train_generated']} | {base['counts']['train_total']} | "
            f"{base['counts']['validation']} |"
        ),
        (
            f"| 原始数据 + GAN 生成数据 | {augmented['counts']['train_original']} | "
            f"{augmented['counts']['train_generated']} | {augmented['counts']['train_total']} | "
            f"{augmented['counts']['validation']} |"
        ),
        "",
        "## 分类效果",
        "",
        "| 实验组 | 最佳验证准确率 | 最终验证准确率 | 训练耗时 |",
        "| --- | ---: | ---: | ---: |",
        (
            f"| 原始数据基线 | {_percent(base['best_val_accuracy'])} | "
            f"{_percent(base['final_val_accuracy'])} | {base['elapsed_seconds']:.2f}s |"
        ),
        (
            f"| 原始数据 + GAN 生成数据 | {_percent(augmented['best_val_accuracy'])} | "
            f"{_percent(augmented['final_val_accuracy'])} | {augmented['elapsed_seconds']:.2f}s |"
        ),
        "",
        "## 对比结论",
        "",
        (
            f"- 本次增强实验额外加入 {extra_samples} 张生成样本，训练集规模增加 "
            f"{total_delta} 张。"
        ),
        (
            f"- 最终验证准确率变化为 {final_delta * 100:+.2f} 个百分点；"
            f"最佳验证准确率变化为 {best_delta * 100:+.2f} 个百分点。"
        ),
        f"- 增强实验训练耗时增加 {time_delta:.2f}s，主要来自训练样本数量增加。",
        (
            "- 从本轮单次实验看，GAN 样本使最终准确率略有提升，但最佳准确率未超过原始基线。"
            "论文中应将其表述为初步有效性验证，并继续通过多随机种子、不同增强比例和更长训练轮数确认稳定性。"
        ),
        "",
        "## 结果文件",
        "",
        f"- 原始基线目录：`{Path(base_dir)}`",
        f"- 增强实验目录：`{Path(augmented_dir)}`",
        f"- 原始基线混淆矩阵：`{Path(base_dir) / 'confusion_matrix.png'}`",
        f"- 增强实验混淆矩阵：`{Path(augmented_dir) / 'confusion_matrix.png'}`",
        "",
    ]
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(description="Build a Markdown report from classifier validation summaries.")
    parser.add_argument("--base-dir", required=True, help="Directory containing baseline summary.json")
    parser.add_argument("--augmented-dir", required=True, help="Directory containing augmented summary.json")
    parser.add_argument("--output", required=True, help="Output Markdown path")
    return parser.parse_args()


def main():
    args = parse_args()
    report = build_classifier_report(args.base_dir, args.augmented_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"Report written to {output_path}")


if __name__ == "__main__":
    main()
