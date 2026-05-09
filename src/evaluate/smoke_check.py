import argparse
import json
from pathlib import Path

import cv2
import torch


MIN_THESIS_CHARS = 15000


REQUIRED_PATHS = [
    "毕业论文_初稿.md",
    "readme.md",
    "docs/defense_summary.md",
    "docs/defense_demo_checklist.md",
    "docs/defense_live_demo_profile.md",
    "docs/reproducibility_manifest.md",
    "docs/classifier_model_comparison_cgan_v2_600_seed42.md",
    "毕业论文_初稿_格式化.docx",
    "src/evaluate/detection_validation.py",
    "src/evaluate/run_classifier_model_comparison.py",
    "src/evaluate/defense_demo_profile.py",
    "assets/figures/pitted_surface_refinement_grid.png",
    "assets/figures/ratio_ablation_cgan_v2_40ep.png",
    "assets/figures/multiseed_cgan_v2_600.png",
    "assets/figures/industrial_gate_comparison.png",
    "assets/figures/system_workflow_industrial.png",
    "results/ratio_ablation/cgan_v2_40ep_seed42/ratio_ablation_summary.csv",
    "results/multiseed/cgan_v2_40ep_600/multiseed_summary.csv",
    "results/industrial_readiness/cgan_v2_40ep_600_seed42/industrial_readiness.json",
]


def _check_path(path):
    path = Path(path)
    if not path.exists():
        return {"name": str(path), "passed": False, "detail": "missing"}
    if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            return {"name": str(path), "passed": False, "detail": "unreadable image"}
        nonblank = float(image.std()) > 1.0
        return {
            "name": str(path),
            "passed": nonblank,
            "detail": f"image {image.shape[1]}x{image.shape[0]}, std={float(image.std()):.2f}",
        }
    return {"name": str(path), "passed": True, "detail": "exists"}


def run_smoke_check(min_thesis_chars=MIN_THESIS_CHARS):
    checks = [_check_path(path) for path in REQUIRED_PATHS]
    thesis_path = Path("毕业论文_初稿.md")
    thesis_chars = len(thesis_path.read_text(encoding="utf-8", errors="replace")) if thesis_path.exists() else 0
    checks.append(
        {
            "name": "thesis_length",
            "passed": thesis_chars >= min_thesis_chars,
            "detail": f"{thesis_chars} chars >= {min_thesis_chars}",
        }
    )
    checks.append(
        {
            "name": "cuda_available",
            "passed": torch.cuda.is_available(),
            "detail": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only",
        }
    )
    passed = all(item["passed"] for item in checks)
    return {"passed": passed, "checks": checks}


def main():
    parser = argparse.ArgumentParser(description="Run a lightweight defense-readiness smoke check.")
    parser.add_argument("--output", default="docs/smoke_check.json")
    parser.add_argument("--min-thesis-chars", type=int, default=MIN_THESIS_CHARS)
    args = parser.parse_args()

    result = run_smoke_check(min_thesis_chars=args.min_thesis_chars)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
