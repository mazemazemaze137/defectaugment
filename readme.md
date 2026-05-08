# DefectAugment 工业表面缺陷数据增强系统

DefectAugment 是一个面向金属表面缺陷图像的毕业设计项目。系统围绕 NEU-DET 数据集，集成数据预处理、传统增强、条件 GAN 生成式增强、生成质量评估、样本筛选和下游分类验证，并通过 Streamlit 提供可视化操作界面。

## 核心功能

- 数据预处理：支持灰度化、尺寸标准化、标注框 ROI 裁剪、CLAHE 对比度增强和中值滤波去噪。
- 传统增强：支持旋转、翻转、亮度对比度扰动和噪声扰动，可按类别目录批量生成样本。
- cGAN 生成式增强：按缺陷类别条件生成样本，采用 Upsample+Conv、LSGAN、谱归一化、AMP 和断点续训。
- 质量评估：支持 SSIM、PSNR、FID，并可从定量角度分析生成样本质量。
- 样本筛选：按清晰度、灰度方差和亮度阈值筛选生成样本，降低低质量样本进入下游训练的概率。
- 下游验证：训练轻量 CNN 分类器，对比原始数据、GAN 增强数据和传统增强数据的验证准确率。

## 快速开始

```powershell
.\.venv\Scripts\activate
pip install -r requirements.txt
streamlit run src/app.py
```

默认界面地址为 `http://127.0.0.1:8501`。

## 常用命令

```powershell
# 运行下游分类基线，并启用早停
python -m src.evaluate.classifier_validation --train-dir data/raw/NEU-DET/train/images --val-dir data/raw/NEU-DET/validation/images --output-dir results/ablation_earlystop/base --epochs 20 --batch-size 16 --image-size 128 --num-workers 0 --early-stopping-patience 4

# 筛选 GAN 生成样本，每类保留 50 张
python -m src.augment.filter_generated_samples --input-dir results/gan_optimized_20ep/export_100_per_class --output-dir results/gan_optimized_20ep/export_filtered_50_per_class --max-per-class 50 --min-sharpness 5 --min-std 8

# 运行筛选后 GAN 样本的下游验证
python -m src.evaluate.classifier_validation --train-dir data/raw/NEU-DET/train/images --val-dir data/raw/NEU-DET/validation/images --generated-dir results/gan_optimized_20ep/export_filtered_50_per_class --output-dir results/ablation_earlystop/gan_filtered_300 --epochs 20 --batch-size 16 --image-size 128 --num-workers 0 --early-stopping-patience 4

# 生成分类验证 Markdown 报告
python -m src.evaluate.experiment_report --base-dir results/ablation_earlystop/base --augmented-dir results/ablation_earlystop/gan_filtered_300 --augmented-name "原始数据 + 筛选后GAN样本" --output results/ablation_earlystop/report_gan_filtered.md
```

## 当前实验结论

在 RTX 3070 + CUDA 环境下，优化后的 20 轮 cGAN 使用 3172 张 ROI 样本、batch size 16、AMP 和内存缓存，耗时约 140.79 秒，吞吐约 450.60 images/s。导出每类 100 张、共 600 张生成样本后，质量评估结果为 SSIM 0.1267、PSNR 12.35 dB、FID 425.86。

启用早停后，20 轮轻量 CNN 下游验证得到如下对照结果：

| 实验组 | 增强样本 | 实际轮数 | 最佳轮数 | 最佳验证准确率 | 最终验证准确率 | 最佳验证损失 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 原始数据基线 | 0 | 9 | 5 | 95.56% | 95.00% | 0.1994 |
| 筛选后 GAN 增强 | 300 | 12 | 8 | 96.11% | 95.56% | 0.1057 |
| cGAN-v2 增强 | 600 | 19 | 15 | 96.39% | 93.06% | 0.1472 |
| 传统增强 | 600 | 18 | 14 | 98.61% | 98.06% | 0.0864 |

新增 cGAN-v2 使用 Projection Discriminator、Hinge Loss、DiffAugment 和 EMA。20 轮 ROI 训练耗时约 245.50 秒，吞吐约 258.42 images/s；导出 600 张样本后的质量指标为 SSIM 0.1529、PSNR 15.14 dB、FID 217.07，相比旧 cGAN 的 FID 425.86 有明显改善。下游分类中，cGAN-v2 的最佳验证准确率达到 96.39%，高于原始基线和旧 GAN 增强，但仍低于传统增强。

实验表明，模型结构升级确实改善了生成样本分布，并带来了更高的最佳分类准确率；但当前 GAN 增强仍未全面超过传统增强。论文中可据此形成较稳妥的结论：本系统不仅实现了生成式增强，还具备评估、筛选和下游对照实验能力；后续应重点优化少数困难类别生成质量、增强比例和多随机种子稳定性。
