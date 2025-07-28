# 🚀 快速开始指南
# Quick Start Guide

## 📋 系统要求

- Python 3.8+
- 8GB+ RAM
- GPU推荐（CUDA支持）
- 10GB+ 磁盘空间

## ⚡ 5分钟快速开始

### 1. 克隆项目
```bash
git clone <repository-url>
cd satellite-roof-detection
```

### 2. 一键初始化
```bash
# 运行初始化脚本
python scripts/setup_project.py
```

这个脚本会：
- ✅ 创建所有必要的目录
- ✅ 安装Python依赖
- ✅ 检查GPU可用性
- ✅ 创建示例notebook

### 3. 配置API密钥
编辑 `config/data_config.yaml`，替换您的Roboflow API密钥：
```yaml
roboflow:
  api_key: "YOUR_API_KEY_HERE"  # 替换为您的密钥
```

### 4. 一键训练
```bash
# 下载数据并开始训练
python scripts/train_model.py --download
```

## 🎯 详细步骤

### 步骤1: 环境设置
```bash
# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 步骤2: 数据准备
```bash
# 方法1: 自动下载
python src/data/download_dataset.py

# 方法2: 手动配置
# 将您的数据放在 data/raw/ 目录下
# 编辑 config/data.yaml 配置路径
```

### 步骤3: 训练模型
```bash
# 基础训练
python scripts/train_model.py

# 自定义参数训练
python scripts/train_model.py --epochs 100 --batch 8 --lr0 0.0001
```

### 步骤4: 查看结果
```bash
# 使用notebook查看详细结果
jupyter notebook notebooks/03_结果分析.ipynb

# 或查看自动生成的图表
ls runs/segment/train_improved/
```

## 📊 预期结果

训练完成后，您应该看到：

### 文件结构
```
runs/segment/train_improved/
├── weights/
│   ├── best.pt          # 最佳模型 ⭐
│   └── last.pt          # 最后模型
├── results.csv          # 训练日志
├── results.png          # 训练曲线
├── confusion_matrix.png # 混淆矩阵
└── 其他分析图表...
```

### 性能指标
- **mAP50**: 期望 > 0.6 (良好), > 0.7 (优秀)
- **各类别F1**: 期望 > 0.5 (可接受)
- **训练时间**: 约1-3小时（取决于硬件）

## 🔧 常见问题

### Q: 训练很慢怎么办？
A: 
- 检查是否使用GPU: `nvidia-smi`
- 减小batch size: `--batch 8`
- 使用更小的模型: 编辑 `config/model_config.yaml`

### Q: 内存不足怎么办？
A:
- 减小batch size: `--batch 4`
- 减小图像尺寸: `--imgsz 512`
- 关闭其他程序

### Q: 数据下载失败？
A:
- 检查网络连接
- 验证API密钥
- 手动下载数据到 `data/raw/`

### Q: 某个类别检测效果差？
A:
- 调整类别权重: 编辑 `config/data_config.yaml`
- 增加该类别的训练数据
- 提高训练轮数

## 📈 进阶使用

### 自定义训练
```bash
# 长时间训练
python scripts/train_model.py --epochs 200 --patience 30

# 使用更大模型
# 编辑 config/model_config.yaml: name: "yolov8l-seg.pt"

# 多GPU训练
python scripts/train_model.py --device 0,1,2,3
```

### 批量预测
```bash
python scripts/predict_batch.py --input_dir test_images/ --output_dir results/
```

### 模型评估
```bash
python scripts/evaluate_model.py --model runs/segment/train_improved/weights/best.pt
```

## 🎯 性能优化建议

### 如果mAP < 0.5
1. 检查数据质量和标注
2. 增加训练轮数到100+
3. 调整类别权重
4. 使用更大的模型

### 如果某类别F1 < 0.3
1. 增加该类别权重到1.5+
2. 添加更多该类别数据
3. 检查该类别标注质量

### 如果训练不收敛
1. 降低学习率到5e-5
2. 增加warmup轮数
3. 检查数据配置

## 📞 获取帮助

- 📖 查看详细文档: `docs/`
- 🐛 报告问题: GitHub Issues
- 💬 讨论交流: GitHub Discussions

## 🎉 成功案例

训练成功后，您的模型应该能够：
- ✅ 准确识别卫星图像中的屋顶
- ✅ 区分不同类型的农田
- ✅ 检测稻田区域
- ✅ 识别裸地区域

恭喜您完成了卫星图像分割检测模型的训练！🎊
