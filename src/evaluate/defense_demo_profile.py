import argparse
import json
from pathlib import Path


DEMO_PROFILE = {
    "name": "DefectAugment defense live demo profile",
    "purpose": "Use short live runs to prove the pipeline, and use saved formal results to prove model quality.",
    "live_demo": {
        "gan_epochs": 3,
        "gan_image_size": 128,
        "gan_save_interval": 1,
        "classifier_epochs": 3,
        "detector_epochs": 1,
        "detector_max_train": 20,
        "detector_max_val": 10,
        "quality_eval_max_pairs": 80,
        "quality_eval_fid": False,
    },
    "formal_results": {
        "cgan_v2_40ep_fid": 178.41,
        "ratio_ablation_best": "600 generated samples, best validation accuracy 99.72%, final validation accuracy 98.61%",
        "multiseed_best": "cGAN-v2 600 samples average best validation accuracy 99.44%, baseline 98.15%",
        "industrial_gate": "600-sample setting passed: min recall 98.33%, cost-weighted error 0.29%",
        "classifier_comparison": "small_cnn 96.11%, mobilenet_v3_small 95.56%, resnet18 93.33% in the formal backbone comparison run",
    },
    "evidence_paths": {
        "demo_checklist": "docs/defense_demo_checklist.md",
        "defense_summary": "docs/defense_summary.md",
        "formatted_thesis": "毕业论文_初稿_格式化.docx",
        "ratio_ablation_figure": "assets/figures/ratio_ablation_cgan_v2_40ep.png",
        "multiseed_figure": "assets/figures/multiseed_cgan_v2_600.png",
        "industrial_gate_figure": "assets/figures/industrial_gate_comparison.png",
        "classifier_comparison": "docs/classifier_model_comparison_cgan_v2_600_seed42.md",
    },
    "talk_track": [
        "现场短轮次训练用于展示数据读取、ROI 预处理、GPU 调用、Loss 曲线、断点续训和文件输出。",
        "正式生成效果不依赖现场随机训练，而是展示 40 轮 cGAN-v2 模型、质量指标、比例消融和多随机种子验证。",
        "若老师要求看完整训练，可说明完整实验已保存 checkpoint、summary、曲线和证据图，现场重新训练会浪费答辩时间。",
        "目标检测验证当前证明 XML 标注读取和检测训练链路可运行；GAN ROI 样本没有 bbox，因此不直接混入检测训练。",
    ],
}


def _path_state(path):
    p = Path(path)
    if p.exists():
        return "exists"
    return "missing"


def build_demo_profile(output_md="docs/defense_live_demo_profile.md", output_json="docs/defense_live_demo_profile.json"):
    output_md = Path(output_md)
    output_json = Path(output_json)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    checked = {
        key: {"path": path, "state": _path_state(path)}
        for key, path in DEMO_PROFILE["evidence_paths"].items()
    }
    payload = {**DEMO_PROFILE, "checked_evidence_paths": checked}

    lines = [
        "# 答辩现场演示配置",
        "",
        "## 现场演示原则",
        "",
        "现场运行短流程，正式结论展示已保存的 40 轮 cGAN-v2 结果、比例消融、多随机种子和工业门槛评估。这样既能证明系统可运行，又能避免把答辩时间消耗在随机训练上。",
        "",
        "## 推荐现场参数",
        "",
        f"- GAN 演示训练轮数：{DEMO_PROFILE['live_demo']['gan_epochs']} 轮",
        f"- GAN 演示分辨率：{DEMO_PROFILE['live_demo']['gan_image_size']}x{DEMO_PROFILE['live_demo']['gan_image_size']}",
        f"- 分类验证演示轮数：{DEMO_PROFILE['live_demo']['classifier_epochs']} 轮",
        f"- 目标检测 smoke 轮数：{DEMO_PROFILE['live_demo']['detector_epochs']} 轮",
        f"- 质量评估演示配对数：{DEMO_PROFILE['live_demo']['quality_eval_max_pairs']}，现场默认不计算 FID",
        "",
        "## 正式结果引用",
        "",
    ]
    for key, value in DEMO_PROFILE["formal_results"].items():
        lines.append(f"- {key}: {value}")

    lines.extend(["", "## 证据路径检查", ""])
    for key, item in checked.items():
        mark = "OK" if item["state"] == "exists" else "MISSING"
        lines.append(f"- {mark} {key}: `{item['path']}`")

    lines.extend(["", "## 答辩表述建议", ""])
    lines.extend(f"- {item}" for item in DEMO_PROFILE["talk_track"])

    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"markdown": str(output_md), "json": str(output_json), "checked": checked}


def main():
    parser = argparse.ArgumentParser(description="Build the defense live-demo profile.")
    parser.add_argument("--output-md", default="docs/defense_live_demo_profile.md")
    parser.add_argument("--output-json", default="docs/defense_live_demo_profile.json")
    args = parser.parse_args()
    result = build_demo_profile(args.output_md, args.output_json)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
