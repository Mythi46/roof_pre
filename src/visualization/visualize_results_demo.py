#!/usr/bin/env python3
"""
可视化结果演示脚本
Visualization Results Demo Script

选取20组图片进行推理并生成可视化结果
"""

import os
import random
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import json
from datetime import datetime

def find_test_images(data_yaml_path, num_images=20):
    """从数据集中选取测试图片"""

    print("🔍 查找测试图片...")

    # 读取数据集配置
    import yaml
    with open(data_yaml_path) as f:
        config = yaml.safe_load(f)

    # 获取数据集根目录
    dataset_root = Path(data_yaml_path).parent

    # 查找验证集图片
    val_images_dir = dataset_root / "val" / "images"
    train_images_dir = dataset_root / "train" / "images"

    # 收集所有图片
    image_files = []

    # 优先使用验证集
    if val_images_dir.exists():
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(list(val_images_dir.glob(ext)))
            image_files.extend(list(val_images_dir.glob(ext.upper())))

    # 如果验证集图片不够，从训练集补充
    if len(image_files) < num_images and train_images_dir.exists():
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            train_files = list(train_images_dir.glob(ext))
            train_files.extend(list(train_images_dir.glob(ext.upper())))
            image_files.extend(train_files)

    if len(image_files) == 0:
        print("❌ 未找到图片文件")
        return []

    # 随机选择指定数量的图片
    selected_images = random.sample(image_files, min(num_images, len(image_files)))

    print(f"✅ 找到 {len(image_files)} 张图片，选择了 {len(selected_images)} 张")

    return selected_images

def load_best_model():
    """加载最佳训练模型"""

    print("🔧 加载最佳模型...")

    # 尝试加载继续训练的最佳模型
    model_paths = [
        "runs/segment/continue_training_optimized/weights/best.pt",
        "runs/segment/improved_training_compatible/weights/best.pt",
        "models/pretrained/yolov8l-seg.pt"
    ]

    for model_path in model_paths:
        if os.path.exists(model_path):
            print(f"✅ 加载模型: {model_path}")
            model = YOLO(model_path)
            return model, model_path

    print("❌ 未找到训练好的模型")
    return None, None

def predict_and_visualize(model, image_path, output_dir, image_index):
    """对单张图片进行推理并可视化"""

    # 执行推理
    results = model.predict(
        source=str(image_path),
        conf=0.35,
        iou=0.6,
        imgsz=896,
        save=False,
        verbose=False
    )[0]

    # 读取原图
    image = cv2.imread(str(image_path))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 生成可视化结果
    annotated_image = results.plot()
    annotated_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

    # 创建对比图
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # 原图
    axes[0].imshow(image_rgb)
    axes[0].set_title(f'原图 {image_index+1}: {image_path.name}', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    # 检测结果
    axes[1].imshow(annotated_rgb)

    # 统计检测结果
    num_detections = len(results.boxes) if results.boxes else 0
    detection_info = []

    if results.boxes:
        class_counts = {}
        for box in results.boxes:
            class_id = int(box.cls[0])
            class_name = results.names[class_id]
            confidence = float(box.conf[0])

            if class_name not in class_counts:
                class_counts[class_name] = 0
            class_counts[class_name] += 1

            detection_info.append({
                'class': class_name,
                'confidence': confidence,
                'bbox': box.xyxy[0].tolist()
            })

        # 生成标题
        title_parts = [f'检测结果 (共{num_detections}个目标)']
        for class_name, count in class_counts.items():
            title_parts.append(f'{class_name}: {count}个')

        title = '\n'.join(title_parts)
    else:
        title = '检测结果 (未检测到目标)'

    axes[1].set_title(title, fontsize=14, fontweight='bold')
    axes[1].axis('off')

    plt.tight_layout()

    # 保存对比图
    output_path = output_dir / f"result_{image_index+1:02d}_{image_path.stem}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 保存单独的检测结果
    result_only_path = output_dir / f"detection_{image_index+1:02d}_{image_path.stem}.jpg"
    cv2.imwrite(str(result_only_path), annotated_image)

    return {
        'image_path': str(image_path),
        'output_path': str(output_path),
        'result_only_path': str(result_only_path),
        'num_detections': num_detections,
        'detections': detection_info,
        'image_size': image.shape[:2]
    }

def create_summary_visualization(results_data, output_dir):
    """创建结果总览可视化"""

    print("📊 创建结果总览...")

    # 统计总体结果
    total_images = len(results_data)
    total_detections = sum(r['num_detections'] for r in results_data)

    # 统计各类别检测数量
    class_counts = {}
    confidence_scores = []

    for result in results_data:
        for detection in result['detections']:
            class_name = detection['class']
            confidence = detection['confidence']

            if class_name not in class_counts:
                class_counts[class_name] = 0
            class_counts[class_name] += 1
            confidence_scores.append(confidence)

    # 创建统计图表
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. 每张图片的检测数量
    image_indices = range(1, total_images + 1)
    detection_counts = [r['num_detections'] for r in results_data]

    axes[0, 0].bar(image_indices, detection_counts, color='skyblue', alpha=0.7)
    axes[0, 0].set_title('每张图片的检测数量', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('图片编号')
    axes[0, 0].set_ylabel('检测数量')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. 类别分布
    if class_counts:
        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

        axes[0, 1].pie(counts, labels=classes, autopct='%1.1f%%',
                       colors=colors[:len(classes)], startangle=90)
        axes[0, 1].set_title('类别分布', fontsize=14, fontweight='bold')

    # 3. 置信度分布
    if confidence_scores:
        axes[1, 0].hist(confidence_scores, bins=20, color='lightgreen', alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('置信度分布', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('置信度')
        axes[1, 0].set_ylabel('频次')
        axes[1, 0].grid(True, alpha=0.3)

        # 添加统计信息
        mean_conf = np.mean(confidence_scores)
        axes[1, 0].axvline(mean_conf, color='red', linestyle='--',
                          label=f'平均置信度: {mean_conf:.3f}')
        axes[1, 0].legend()

    # 4. 总体统计
    axes[1, 1].axis('off')
    stats_text = f"""
📊 检测结果总览

🖼️ 总图片数: {total_images}
🎯 总检测数: {total_detections}
📈 平均每图: {total_detections/total_images:.1f}个目标

📋 类别统计:
"""

    for class_name, count in class_counts.items():
        percentage = (count / total_detections) * 100 if total_detections > 0 else 0
        stats_text += f"   {class_name}: {count}个 ({percentage:.1f}%)\n"

    if confidence_scores:
        stats_text += f"\n🎯 置信度统计:\n"
        stats_text += f"   平均: {np.mean(confidence_scores):.3f}\n"
        stats_text += f"   最高: {np.max(confidence_scores):.3f}\n"
        stats_text += f"   最低: {np.min(confidence_scores):.3f}\n"

    axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes,
                    fontsize=12, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))

    plt.suptitle('🏠 屋顶检测结果总览 - 20张图片可视化分析', fontsize=16, fontweight='bold')
    plt.tight_layout()

    # 保存总览图
    summary_path = output_dir / "detection_summary.png"
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    plt.close()

    return str(summary_path)

def main():
    """主函数"""

    print("🎨 屋顶检测可视化演示")
    print("=" * 50)

    # 创建输出目录
    output_dir = Path("visualization_results")
    output_dir.mkdir(exist_ok=True)

    # 数据集配置
    data_yaml = "data/raw/new-2-1/data.yaml"

    if not os.path.exists(data_yaml):
        print(f"❌ 数据集配置文件不存在: {data_yaml}")
        return

    # 加载模型
    model, model_path = load_best_model()
    if model is None:
        return

    # 选择测试图片
    test_images = find_test_images(data_yaml, num_images=20)
    if not test_images:
        return

    print(f"\n🚀 开始处理 {len(test_images)} 张图片...")

    # 处理每张图片
    results_data = []

    for i, image_path in enumerate(test_images):
        print(f"🔍 处理图片 {i+1}/20: {image_path.name}")

        try:
            result = predict_and_visualize(model, image_path, output_dir, i)
            results_data.append(result)
            print(f"   ✅ 检测到 {result['num_detections']} 个目标")

        except Exception as e:
            print(f"   ❌ 处理失败: {e}")
            continue

    # 创建总览
    if results_data:
        summary_path = create_summary_visualization(results_data, output_dir)

        # 保存详细结果
        results_json = {
            'model_path': model_path,
            'timestamp': datetime.now().isoformat(),
            'total_images': len(results_data),
            'total_detections': sum(r['num_detections'] for r in results_data),
            'results': results_data
        }

        json_path = output_dir / "detection_results.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results_json, f, indent=2, ensure_ascii=False)

        print(f"\n🎉 可视化完成!")
        print(f"📁 结果保存在: {output_dir}")
        print(f"📊 总览图: {summary_path}")
        print(f"📋 详细结果: {json_path}")
        print(f"🖼️ 处理图片: {len(results_data)}/20")
        print(f"🎯 总检测数: {sum(r['num_detections'] for r in results_data)}")

        # 显示类别统计
        class_counts = {}
        for result in results_data:
            for detection in result['detections']:
                class_name = detection['class']
                if class_name not in class_counts:
                    class_counts[class_name] = 0
                class_counts[class_name] += 1

        print(f"\n📋 类别检测统计:")
        for class_name, count in class_counts.items():
            print(f"   {class_name}: {count}个")

    else:
        print("❌ 没有成功处理任何图片")

if __name__ == "__main__":
    main()