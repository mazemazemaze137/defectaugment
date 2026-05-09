import argparse
import csv
import json
from pathlib import Path


def _read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _pct(value):
    return f"{float(value) * 100:.2f}%"


def _load_optional_json(path):
    path = Path(path)
    if not path.exists():
        return None
    return _read_json(path)


def build_summary(output):
    ratio_rows = _read_csv("results/ratio_ablation/cgan_v2_40ep_multiseed/ratio_ablation_summary.csv")
    multiseed_rows = _read_csv("results/multiseed/cgan_v2_40ep_600/multiseed_summary.csv")
    multiseed_runs = _read_csv("results/multiseed/cgan_v2_40ep_600/multiseed_runs.csv")
    industrial_300 = _load_optional_json("results/industrial_readiness/cgan_v2_40ep/industrial_readiness.json")
    industrial_600 = _load_optional_json("results/industrial_readiness/cgan_v2_40ep_600_seed42/industrial_readiness.json")

    lines = [
        "# DefectAugment 答辩实验汇总",
        "",
        "## 一句话结论",
        "",
        "40 轮 cGAN-v2 结合质量筛选和合适生成比例后，能够提升 NEU-DET 六类缺陷分类的最佳验证准确率；3 随机种子比例消融显示每类 50 张方案平均最佳准确率最高，每类 100 张方案在 seed 42 和工业应用门槛中表现较强，但对随机种子更敏感。",
        "",
        "## 生成比例消融",
        "",
        "| 每类生成样本 | 生成样本总数 | 种子数 | 平均最佳准确率 | 标准差 | 平均最终准确率 | 平均最佳损失 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(ratio_rows, key=lambda item: int(item["samples_per_class"])):
        std = float(row["best_val_accuracy_std"]) if row.get("best_val_accuracy_std") else 0.0
        lines.append(
            f"| {int(row['samples_per_class'])} | {int(row['generated_total'])} | "
            f"{int(row['seeds'])} | {_pct(row['best_val_accuracy_mean'])} | {std * 100:.2f}% | "
            f"{_pct(row['final_val_accuracy_mean'])} | "
            f"{float(row['best_val_loss_mean']):.4f} |"
        )

    lines.extend(
        [
            "",
            "## 600 张方案多随机种子复验",
            "",
            "| 实验组 | 种子数 | 平均最佳准确率 | 标准差 | 平均最终准确率 | 平均最佳损失 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sorted(multiseed_rows, key=lambda item: item["group"]):
        lines.append(
            f"| {row['group']} | {int(row['seeds'])} | {_pct(row['best_val_accuracy_mean'])} | "
            f"{_pct(row['best_val_accuracy_std'])} | {_pct(row['final_val_accuracy_mean'])} | "
            f"{float(row['best_val_loss_mean']):.4f} |"
        )
    base = next(row for row in multiseed_rows if row["group"] == "base")
    augmented = next(row for row in multiseed_rows if row["group"] == "augmented")
    delta = float(augmented["best_val_accuracy_mean"]) - float(base["best_val_accuracy_mean"])
    loss_delta = float(base["best_val_loss_mean"]) - float(augmented["best_val_loss_mean"])
    lines.extend(
        [
            "",
            f"- 600 张 cGAN-v2 增强组平均最佳准确率相对基线提升 {delta * 100:.2f} 个百分点。",
            f"- 平均最佳验证损失降低 {loss_delta:.4f}，说明增强样本不仅提升准确率，也改善了收敛质量。",
            "",
            "| Seed | 基线最佳准确率 | 600张增强最佳准确率 | 基线最佳损失 | 600张增强最佳损失 |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    seeds = sorted({int(row["seed"]) for row in multiseed_runs})
    for seed in seeds:
        base_run = next(row for row in multiseed_runs if row["group"] == "base" and int(row["seed"]) == seed)
        aug_run = next(row for row in multiseed_runs if row["group"] == "augmented" and int(row["seed"]) == seed)
        lines.append(
            f"| {seed} | {_pct(base_run['best_val_accuracy'])} | {_pct(aug_run['best_val_accuracy'])} | "
            f"{float(base_run['best_val_loss']):.4f} | {float(aug_run['best_val_loss']):.4f} |"
        )

    if industrial_300 and industrial_600:
        lines.extend(
            [
                "",
                "## 工业应用就绪度",
                "",
                "| 方案 | 最佳验证准确率 | 最低类别召回率 | 代价加权错误率 | 是否通过 |",
                "| --- | ---: | ---: | ---: | --- |",
                f"| 300 张 cGAN-v2 | {_pct(industrial_300['best_val_accuracy'])} | {_pct(industrial_300['min_class_recall'])} | {_pct(industrial_300['weighted_error_rate'])} | {'通过' if industrial_300['passed'] else '未通过'} |",
                f"| 600 张 cGAN-v2 | {_pct(industrial_600['best_val_accuracy'])} | {_pct(industrial_600['min_class_recall'])} | {_pct(industrial_600['weighted_error_rate'])} | {'通过' if industrial_600['passed'] else '未通过'} |",
            ]
        )

    lines.extend(
        [
            "",
            "## 答辩可讲重点",
            "",
            "1. 系统不是只做 GAN 生成，而是形成预处理、生成、质量评估、筛选、下游验证、工业门槛判断的闭环。",
            "2. 早期 300 张方案能提升准确率但困难类别召回不足，因此不直接声明可上线。",
            "3. 通过 3 随机种子比例消融发现，每类 50 张的平均最佳准确率最高，600 张方案在工业门槛中表现强但随机性更明显。",
            "4. 600 张方案的多随机种子复验显示增强组平均最佳准确率和平均最佳损失均优于基线，可作为工业试运行候选方案。",
            "",
            "## 可引用图表",
            "",
            "- `assets/figures/pitted_surface_refinement_grid.png`",
            "- `assets/figures/ratio_ablation_cgan_v2_40ep.png`",
            "- `assets/figures/multiseed_cgan_v2_600.png`",
            "- `assets/figures/industrial_gate_comparison.png`",
            "- `assets/figures/system_workflow_industrial.png`",
            "",
            "## 复现材料",
            "",
            "- `docs/reproducibility_manifest.md`：记录 Python、CUDA、PyTorch、依赖版本、Git 提交和关键数据路径。",
            "- `docs/defense_demo_checklist.md`：记录答辩演示顺序、常见追问和谨慎表述。",
        ]
    )

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser(description="Build a thesis defense summary from experiment outputs.")
    parser.add_argument("--output", default="docs/defense_summary.md")
    args = parser.parse_args()
    output = build_summary(args.output)
    print(json.dumps({"output": str(output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
