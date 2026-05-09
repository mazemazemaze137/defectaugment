import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path


PACKAGES = [
    "torch",
    "torchvision",
    "opencv-python",
    "scikit-image",
    "albumentations",
    "matplotlib",
    "numpy",
    "pandas",
    "streamlit",
    "pytorch-fid",
]


def _run(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError:
        return ""
    return (result.stdout or result.stderr).strip()


def _package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def _torch_info():
    try:
        import torch
    except Exception as exc:
        return {"available": False, "error": str(exc)}

    info = {
        "available": True,
        "version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device_count": torch.cuda.device_count(),
        "devices": [],
    }
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            info["devices"].append(
                {
                    "index": index,
                    "name": props.name,
                    "total_memory_gb": round(props.total_memory / (1024 ** 3), 2),
                    "major": props.major,
                    "minor": props.minor,
                }
            )
    return info


def build_manifest(output):
    output = Path(output)
    manifest = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "git_commit": _run(["git", "rev-parse", "--short", "HEAD"]),
        "git_branch": _run(["git", "branch", "--show-current"]),
        "torch": _torch_info(),
        "packages": {name: _package_version(name) for name in PACKAGES},
        "key_paths": {
            "raw_train": "data/raw/NEU-DET/train/images",
            "raw_validation": "data/raw/NEU-DET/validation/images",
            "cgan_v2_40ep_generated": "results/cgan_v2_roi_40ep/export_100_per_class",
            "ratio_ablation": "results/ratio_ablation/cgan_v2_40ep_multiseed",
            "multiseed_600": "results/multiseed/cgan_v2_40ep_600",
            "figures": "assets/figures",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# DefectAugment 复现环境清单",
        "",
        "## 系统信息",
        "",
        f"- Python：`{manifest['python']}`",
        f"- 平台：`{manifest['platform']}`",
        f"- 处理器：`{manifest['processor']}`",
        f"- Git 分支：`{manifest['git_branch']}`",
        f"- Git 提交：`{manifest['git_commit']}`",
        "",
        "## CUDA / PyTorch",
        "",
    ]
    torch_info = manifest["torch"]
    if torch_info.get("available"):
        lines.extend(
            [
                f"- PyTorch：`{torch_info['version']}`",
                f"- CUDA 可用：`{torch_info['cuda_available']}`",
                f"- CUDA 版本：`{torch_info['cuda_version']}`",
                f"- cuDNN 版本：`{torch_info['cudnn_version']}`",
                f"- GPU 数量：`{torch_info['device_count']}`",
            ]
        )
        for device in torch_info["devices"]:
            lines.append(
                f"- GPU {device['index']}：`{device['name']}`，显存约 `{device['total_memory_gb']} GB`，"
                f"计算能力 `{device['major']}.{device['minor']}`"
            )
    else:
        lines.append(f"- PyTorch 不可用：`{torch_info.get('error', 'unknown')}`")

    lines.extend(["", "## 关键依赖", "", "| 包 | 版本 |", "| --- | --- |"])
    for name, version in manifest["packages"].items():
        lines.append(f"| {name} | `{version}` |")
    lines.extend(
        [
            "",
            "说明：当前环境使用 `opencv-python 4.12.0.88` 与 `numpy 2.2.6`，因此 `requirements.txt` 将 NumPy 约束为 `numpy>=2,<2.3`，避免重新安装时与 OpenCV 依赖范围冲突。",
        ]
    )

    lines.extend(["", "## 关键路径", "", "| 项目内容 | 路径 |", "| --- | --- |"])
    for name, path in manifest["key_paths"].items():
        lines.append(f"| {name} | `{path}` |")

    lines.extend(
        [
            "",
            "## 推荐复现实验命令",
            "",
            "```powershell",
            "python -m src.evaluate.run_ratio_ablation --train-dir data/raw/NEU-DET/train/images --val-dir data/raw/NEU-DET/validation/images --generated-source-dir results/cgan_v2_roi_40ep/export_100_per_class --output-dir results/ratio_ablation/cgan_v2_40ep_multiseed --samples-per-class 25 50 100 --seeds 42 7 123 --epochs 20 --batch-size 16 --image-size 128 --early-stopping-patience 4",
            "python -m src.evaluate.run_multiseed_validation --train-dir data/raw/NEU-DET/train/images --val-dir data/raw/NEU-DET/validation/images --generated-dir results/ratio_ablation/cgan_v2_40ep_seed42/subsets/gan_100_per_class --output-dir results/multiseed/cgan_v2_40ep_600 --seeds 42 7 123 --epochs 20 --batch-size 16 --image-size 128 --early-stopping-patience 4 --industrial-report",
            "python -m src.evaluate.make_evidence_figures --output-dir assets/figures",
            "python -m src.evaluate.build_defense_summary --output docs/defense_summary.md",
            "```",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser(description="Build a reproducibility manifest for thesis defense.")
    parser.add_argument("--output", default="docs/reproducibility_manifest.md")
    args = parser.parse_args()
    output = build_manifest(args.output)
    print(json.dumps({"output": str(output), "json": str(output.with_suffix(".json"))}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
