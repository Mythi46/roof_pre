# 🏠 屋顶检测项目 (Roof Detection Project)

## 📋 项目概述

这是一个基于YOLOv8的屋顶分割检测项目，专门用于识别和分割航拍图像中的不同类型屋顶。

### 🎯 检测类别
- `Baren-Land`: 裸地
- `farm`: 农田  
- `rice-fields`: 稻田
- `roof`: 屋顶

## 📁 项目结构

```
roof-detection/
├── data/                    # 数据目录
│   ├── raw/                # 原始数据
│   └── processed/          # 处理后的数据
├── models/                 # 模型目录
│   ├── pretrained/        # 预训练模型
│   └── trained/           # 训练好的模型
├── results/               # 结果目录
│   ├── training/         # 训练结果
│   ├── evaluation/       # 评估结果
│   └── visualization/    # 可视化结果
├── scripts/              # 脚本目录
├── config/               # 配置文件
├── notebooks/            # Jupyter笔记本
└── archive/              # 归档文件
```

## 🚀 快速开始

### 1. 环境设置
```bash
pip install -r requirements.txt
```

### 2. 数据准备
数据集已包含在 `data/raw/new-2-1/` 目录中。

### 3. 开始训练
```bash
python train_expert_correct_solution.py
```

### 4. 可视化结果
```bash
python generate_visualization_results.py
```

## 📊 训练配置

当前使用的专家改进配置：
- **模型**: YOLOv8m-seg (分割模型)
- **图像尺寸**: 768x768
- **批次大小**: 16
- **学习率**: 0.005 (AdamW优化器)
- **损失权重**: cls=1.0, box=7.5, dfl=1.5
- **数据增强**: copy_paste=0.5, mosaic=0.3, mixup=0.1

## 🔧 重要说明

### YOLOv8类别权重问题
⚠️ **重要发现**: YOLOv8不支持`class_weights`参数！

**解决方案**:
1. 调整损失函数权重 (cls, box, dfl)
2. 使用数据增强策略平衡类别
3. 优化训练策略 (学习率、epochs等)

## 📈 性能优化

- 使用余弦退火学习率调度
- 增加分类损失权重以改善类别不平衡
- 针对性数据增强策略
- 早停机制防止过拟合

## 📝 更新日志

- **2024-07-29**: 项目重构和清理
- **2024-07-29**: 修复YOLOv8类别权重问题
- **2024-07-29**: 优化训练配置和数据增强策略

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个项目！

## 📄 许可证

MIT License
