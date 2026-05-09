# DefectAugment 复现环境清单

## 系统信息

- Python：`3.12.6 (tags/v3.12.6:a4a2d2b, Sep  6 2024, 20:11:23) [MSC v.1940 64 bit (AMD64)]`
- 平台：`Windows-11-10.0.26100-SP0`
- 处理器：`AMD64 Family 25 Model 33 Stepping 2, AuthenticAMD`
- Git 分支：`defense-eval-extensions`
- Git 提交：`e8c117c`

## CUDA / PyTorch

- PyTorch：`2.7.1+cu118`
- CUDA 可用：`True`
- CUDA 版本：`11.8`
- cuDNN 版本：`90100`
- GPU 数量：`1`
- GPU 0：`NVIDIA GeForce RTX 3070`，显存约 `8.0 GB`，计算能力 `8.6`

## 关键依赖

| 包 | 版本 |
| --- | --- |
| torch | `2.7.1+cu118` |
| torchvision | `0.22.1+cu118` |
| opencv-python | `4.12.0.88` |
| scikit-image | `0.26.0` |
| albumentations | `2.0.8` |
| matplotlib | `3.10.8` |
| numpy | `2.2.6` |
| pandas | `2.3.3` |
| streamlit | `1.52.2` |
| pytorch-fid | `0.3.0` |

说明：当前环境使用 `opencv-python 4.12.0.88` 与 `numpy 2.2.6`，因此 `requirements.txt` 将 NumPy 约束为 `numpy>=2,<2.3`，避免重新安装时与 OpenCV 依赖范围冲突。

## 关键路径

| 项目内容 | 路径 |
| --- | --- |
| raw_train | `data/raw/NEU-DET/train/images` |
| raw_validation | `data/raw/NEU-DET/validation/images` |
| cgan_v2_40ep_generated | `results/cgan_v2_roi_40ep/export_100_per_class` |
| ratio_ablation | `results/ratio_ablation/cgan_v2_40ep_multiseed` |
| multiseed_600 | `results/multiseed/cgan_v2_40ep_600` |
| figures | `assets/figures` |

## 推荐复现实验命令

```powershell
python -m src.evaluate.run_ratio_ablation --train-dir data/raw/NEU-DET/train/images --val-dir data/raw/NEU-DET/validation/images --generated-source-dir results/cgan_v2_roi_40ep/export_100_per_class --output-dir results/ratio_ablation/cgan_v2_40ep_multiseed --samples-per-class 25 50 100 --seeds 42 7 123 --epochs 20 --batch-size 16 --image-size 128 --early-stopping-patience 4
python -m src.evaluate.run_multiseed_validation --train-dir data/raw/NEU-DET/train/images --val-dir data/raw/NEU-DET/validation/images --generated-dir results/ratio_ablation/cgan_v2_40ep_seed42/subsets/gan_100_per_class --output-dir results/multiseed/cgan_v2_40ep_600 --seeds 42 7 123 --epochs 20 --batch-size 16 --image-size 128 --early-stopping-patience 4 --industrial-report
python -m src.evaluate.make_evidence_figures --output-dir assets/figures
python -m src.evaluate.build_defense_summary --output docs/defense_summary.md
```
