# 🛰️ 卫星图像分割数据集

## 📊 数据集概述
- **名称**: satellite-image-segmentation
- **描述**: 卫星图像分割数据集 - 包含建筑物、道路、植被等
- **格式**: YOLOv8 Segmentation
- **类别数**: 4

## 🏷️ 类别说明
- **0: Baren-Land** - 裸地、空地
- **1: farm** - 农田、耕地
- **2: rice-fields** - 稻田、水田
- **3: roof** - 建筑物屋顶

## 📁 目录结构
```
new-2-1/
├── data.yaml              # YOLOv8配置文件
├── README.md              # 本文件
├── train/                 # 训练集
│   ├── images/           # 训练图像 (.jpg, .png)
│   └── labels/           # 训练标签 (.txt, YOLOv8分割格式)
├── val/                   # 验证集
│   ├── images/           # 验证图像
│   └── labels/           # 验证标签
└── test/                  # 测试集
    ├── images/           # 测试图像
    └── labels/           # 测试标签
```

## 🔧 标签格式
YOLOv8分割格式，每行包含：
```
class_id x1 y1 x2 y2 x3 y3 ... xn yn
```
- `class_id`: 类别ID (0-3)
- `x1 y1 x2 y2 ...`: 归一化的多边形坐标点 (0-1范围)

## 📊 数据集统计
- **训练集**: ~8,000-12,000 张图像
- **验证集**: ~200-500 张图像  
- **测试集**: ~200-500 张图像
- **图像尺寸**: 建议768x768或更高
- **类别分布**: 相对平衡

## 🚀 使用方法

### 训练模型
```python
from ultralytics import YOLO

# 加载预训练模型
model = YOLO('yolov8m-seg.pt')

# 训练
results = model.train(
    data='data/raw/new-2-1/data.yaml',
    epochs=60,
    imgsz=768,
    batch=16,
    device='auto'
)
```

### 专家改进训练
```bash
python train_expert_correct_solution.py
```

## 📋 数据准备步骤

### 1. 获取数据集
- 访问 [Roboflow Universe](https://universe.roboflow.com/)
- 搜索 "satellite segmentation" 或 "aerial segmentation"
- 选择合适的数据集
- 下载YOLOv8格式

### 2. 推荐数据集
- **Aerial Semantic Segmentation Drone Dataset**
- **Satellite Image Segmentation**
- **Building Segmentation Dataset**
- **Agricultural Field Segmentation**

### 3. 数据放置
1. 解压下载的数据集
2. 将文件放入对应目录:
   - 图像 → `train/images/`, `val/images/`, `test/images/`
   - 标签 → `train/labels/`, `val/labels/`, `test/labels/`
3. 确保data.yaml配置正确

### 4. 验证数据集
```python
import yaml
from pathlib import Path

# 检查配置
with open('data/raw/new-2-1/data.yaml', 'r') as f:
    config = yaml.safe_load(f)
    
print(f"类别数: {config['nc']}")
print(f"类别名: {config['names']}")

# 检查文件数量
train_images = len(list(Path('data/raw/new-2-1/train/images').glob('*')))
train_labels = len(list(Path('data/raw/new-2-1/train/labels').glob('*')))
print(f"训练图像: {train_images}, 训练标签: {train_labels}")
```

## ⚠️ 注意事项

1. **标签格式**: 确保使用YOLOv8分割格式，不是检测格式
2. **文件对应**: 每个图像文件必须有对应的标签文件
3. **坐标归一化**: 所有坐标必须在0-1范围内
4. **类别ID**: 确保类别ID与data.yaml中的定义一致
5. **图像质量**: 建议使用高分辨率卫星图像

## 🎯 性能目标

基于专家改进方案，预期性能:
- **mAP50**: >0.92
- **Recall**: >0.88
- **Precision**: >0.90
- **训练稳定性**: 平滑收敛

## 📞 支持

如果遇到问题:
1. 检查数据集格式是否正确
2. 验证文件路径和命名
3. 确认类别定义一致
4. 查看训练日志错误信息

---

**数据集状态**: 🔄 需要手动下载和配置  
**最后更新**: download_satellite_dataset_roboflow.py 生成  
**兼容性**: YOLOv8 分割任务  
