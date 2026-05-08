# DefectAugment 工业表面缺陷数据增强系统

DefectAugment 是一个面向金属表面缺陷数据增强的毕业设计项目。系统基于 Streamlit 提供可视化界面，围绕 NEU-DET 数据集完成数据预处理、传统增强、条件 GAN 生成式增强、生成质量评估和下游分类验证。

## 核心功能

- 数据预处理：支持灰度化、尺寸标准化、标注框 ROI 裁剪、CLAHE 对比度增强和中值滤波去噪。
- 传统增强：支持旋转、翻转等基础几何变换，可批量生成增强样本。
- cGAN 生成式增强：按缺陷类别条件生成样本，采用谱归一化、Upsample+Conv 和 LSGAN 损失，支持 CUDA 训练、断点续训、暂停/继续/停止和损失曲线监控。
- 质量评估：支持 SSIM、PSNR、FID 指标，便于从定量角度分析生成样本质量。
- 下游验证：训练轻量 CNN 分类器，对比“原始数据”和“原始数据 + 生成数据”的验证准确率。

## 快速开始

```powershell
.\.venv\Scripts\activate
pip install -r requirements.txt
streamlit run src/app.py
```

默认界面地址为 `http://127.0.0.1:8501`。

## 推荐目录

- `data/raw/NEU-DET/`：原始 NEU-DET 数据集。
- `data/processed/`：预处理后的训练样本，运行时自动生成。
- `results/`：GAN 输出、质量评估结果、分类验证结果和图表。
- `src/augment/`：传统增强与 GAN 模型代码。
- `src/evaluate/`：质量指标、分类验证和实验报告脚本。

## 常用命令

```powershell
# 启动 Streamlit
streamlit run src/app.py

# 运行下游分类基线
python -m src.evaluate.classifier_validation --train-dir data/raw/NEU-DET/train/images --val-dir data/raw/NEU-DET/validation/images --output-dir results/classifier_validation/base --epochs 10 --batch-size 32 --image-size 128 --num-workers 0

# 运行加入 GAN 样本后的分类验证
python -m src.evaluate.classifier_validation --train-dir data/raw/NEU-DET/train/images --val-dir data/raw/NEU-DET/validation/images --generated-dir results/gan_run_20260402_233902 --output-dir results/classifier_validation/augmented --epochs 10 --batch-size 32 --image-size 128 --num-workers 0

# 生成分类验证 Markdown 报告
python -m src.evaluate.experiment_report --base-dir results/classifier_validation/base --augmented-dir results/classifier_validation/augmented --output results/classifier_validation/report.md

# 从 cGAN 检查点导出按类别组织的正式生成样本
python -m src.augment.export_cgan_samples --checkpoint results/gan_run_xxx/checkpoint_latest.pth --output-dir results/gan_run_xxx/export_100_per_class --samples-per-class 100 --image-size 128
```

## 当前实验结论

在 RTX 3070 + CUDA 环境下，10 轮轻量 CNN 分类验证显示：原始数据基线最终验证准确率为 96.39%，加入 576 张 GAN 生成样本后最终验证准确率为 96.94%。单次实验表明生成样本对最终准确率有轻微提升，但最佳准确率仍需通过多随机种子、更长训练轮数和不同增强比例继续验证。

优化后的 20 轮 cGAN 实验使用 3172 张 ROI 样本、batch size 16、AMP 和内存缓存，耗时约 140.79 秒，吞吐约 450.60 images/s。导出每类 100 张、共 600 张生成样本后，15 轮下游分类验证显示最终验证准确率由 83.33% 提升到 96.94%，但最佳验证准确率由 99.17% 小幅下降到 98.61%，说明当前生成样本更适合作为训练正则化补充，生成质量仍需继续提升。
