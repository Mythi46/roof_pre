"""
模型训练模块 - 专家改进版
Model training module - Expert improved version

支持专家改进的所有功能：
- 自动类别权重计算
- 统一解像度
- 余弦退火学习率
- 分割友好数据增强
"""

import os
import glob
import yaml
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Any, List
from collections import Counter
from ultralytics import YOLO

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RoofDetectionTrainer:
    """屋顶检测模型训练器 - 专家改进版"""

    def __init__(self, config_path: str = "config/model_config.yaml"):
        """
        初始化训练器

        Args:
            config_path: 模型配置文件路径
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.model = None
        self.class_weights = None
        self.class_names = None
        
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"配置文件加载成功: {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"配置文件加载失败: {e}")
            raise
    
    def calculate_class_weights(self, data_yaml: str) -> np.ndarray:
        """
        专家改进1: 自动计算类别权重
        使用有效样本数方法 (Cui et al., 2019)

        Args:
            data_yaml: 数据配置文件路径

        Returns:
            计算得到的类别权重数组
        """
        try:
            # 读取数据配置
            with open(data_yaml, 'r') as f:
                data_config = yaml.safe_load(f)

            self.class_names = data_config['names']
            num_classes = data_config['nc']

            # 获取训练数据路径
            train_path = data_config['train']
            if os.path.isfile(train_path):
                train_dir = os.path.dirname(train_path)
            else:
                train_dir = train_path

            # 查找标签文件
            label_dir = os.path.join(os.path.dirname(train_dir), 'labels') if 'images' in train_dir else os.path.join(train_dir, 'labels')
            label_files = glob.glob(os.path.join(label_dir, '*.txt'))

            if not label_files:
                logger.warning("未找到标签文件，使用默认权重")
                return np.ones(num_classes)

            # 统计各类别实例数
            counter = Counter()
            for f in label_files:
                with open(f) as r:
                    for line in r:
                        if line.strip():
                            cls_id = int(line.split()[0])
                            counter[cls_id] += 1

            logger.info("📊 类别分布统计:")
            for i in range(num_classes):
                count = counter.get(i, 0)
                logger.info(f"   {self.class_names[i]:12}: {count:6d} 个实例")

            # 有效样本数方法计算权重
            beta = 0.999
            freq = np.array([counter.get(i, 0) for i in range(num_classes)], dtype=float)
            freq = np.maximum(freq, 1)  # 避免除零

            eff_num = 1 - np.power(beta, freq)
            cls_weights = (1 - beta) / eff_num
            cls_weights = cls_weights / cls_weights.mean()  # 归一化

            logger.info("🎯 自动计算的类别权重:")
            for i, (name, weight) in enumerate(zip(self.class_names, cls_weights)):
                logger.info(f"   {name:12}: {weight:.3f}")

            self.class_weights = cls_weights
            return cls_weights

        except Exception as e:
            logger.error(f"类别权重计算失败: {e}")
            return np.ones(num_classes)

    def load_model(self, model_path: Optional[str] = None) -> None:
        """
        加载模型

        Args:
            model_path: 模型路径，如果为None则使用配置中的模型
        """
        try:
            if model_path is None:
                model_path = self.config['model']['name']

            self.model = YOLO(model_path)
            logger.info(f"模型加载成功: {model_path}")

        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    def train(
        self,
        data_yaml: str = "config/data.yaml",
        project: str = "runs/segment",
        name: str = "train_expert_local",
        use_expert_improvements: bool = True,
        **kwargs
    ) -> Any:
        """
        训练模型 - 专家改进版

        Args:
            data_yaml: 数据配置文件路径
            project: 项目保存路径
            name: 训练任务名称
            use_expert_improvements: 是否使用专家改进
            **kwargs: 其他训练参数

        Returns:
            训练结果
        """
        if self.model is None:
            self.load_model()

        # 获取训练配置
        train_config = self.config['training'].copy()

        # 专家改进配置
        if use_expert_improvements:
            logger.info("🎯 启用专家改进功能...")

            # 专家改进1: 自动计算类别权重
            class_weights = self.calculate_class_weights(data_yaml)
            train_config['class_weights'] = class_weights.tolist()

            # 专家改进2: 统一解像度
            IMG_SIZE = 768
            train_config['imgsz'] = IMG_SIZE
            logger.info(f"📐 统一解像度: {IMG_SIZE}x{IMG_SIZE}")

            # 专家改进3: 现代学习率策略
            train_config.update({
                'optimizer': 'AdamW',
                'lr0': 2e-4,
                'cos_lr': True,
                'warmup_epochs': 5,
                'warmup_momentum': 0.8,
                'warmup_bias_lr': 0.1,
            })
            logger.info("🔄 启用余弦退火 + AdamW")

            # 专家改进4: 分割友好数据增强
            train_config.update({
                'mosaic': 0.25,      # 大幅降低
                'copy_paste': 0.5,   # 分割专用增强
                'close_mosaic': 0,   # 不延迟关闭
                'mixup': 0.0,
                'hsv_h': 0.02,
                'hsv_s': 0.7,
                'hsv_v': 0.4,
                'degrees': 10.0,
                'translate': 0.1,
                'scale': 0.5,
                'shear': 0.0,
                'perspective': 0.0,
                'flipud': 0.5,
                'fliplr': 0.5,
            })
            logger.info("🎨 启用分割友好数据增强")

            # 专家改进5: 训练控制优化
            train_config.update({
                'patience': 20,
                'save_period': -1,
                'amp': True,
            })

            logger.info("✅ 专家改进配置完成")

        # 应用用户自定义参数
        train_config.update(kwargs)

        # 确保数据配置文件存在
        if not os.path.exists(data_yaml):
            logger.error(f"数据配置文件不存在: {data_yaml}")
            raise FileNotFoundError(f"数据配置文件不存在: {data_yaml}")

        try:
            logger.info("🚀 开始专家改进版训练...")
            logger.info(f"数据配置: {data_yaml}")
            logger.info(f"项目路径: {project}/{name}")

            if use_expert_improvements:
                logger.info("🎯 专家改进摘要:")
                logger.info(f"   自动类别权重: {[f'{w:.3f}' for w in class_weights]}")
                logger.info(f"   统一解像度: {train_config['imgsz']}")
                logger.info(f"   学习率策略: {train_config['optimizer']} + 余弦退火")
                logger.info(f"   数据增强: Mosaic={train_config['mosaic']}, Copy-Paste={train_config['copy_paste']}")

            # 开始训练
            results = self.model.train(
                data=data_yaml,
                project=project,
                name=name,
                **train_config
            )

            logger.info("🎉 专家改进版训练完成!")
            return results

        except Exception as e:
            logger.error(f"训练失败: {e}")
            raise
    
    def get_best_model_path(self, project: str = "runs/segment", name: str = "train_improved") -> str:
        """获取最佳模型路径"""
        return os.path.join(project, name, "weights", "best.pt")
    
    def validate_training_setup(self, data_yaml: str = "config/data.yaml") -> bool:
        """
        验证训练设置
        
        Args:
            data_yaml: 数据配置文件路径
            
        Returns:
            是否验证通过
        """
        try:
            # 检查数据配置文件
            if not os.path.exists(data_yaml):
                logger.error(f"数据配置文件不存在: {data_yaml}")
                return False
            
            # 检查数据配置内容
            with open(data_yaml, 'r') as f:
                data_config = yaml.safe_load(f)
            
            # 检查必要字段
            required_fields = ['train', 'val', 'nc', 'names']
            for field in required_fields:
                if field not in data_config:
                    logger.error(f"数据配置缺少必要字段: {field}")
                    return False
            
            # 检查类别权重（关键改进）
            if 'class_weights' not in data_config:
                logger.warning("数据配置中未找到class_weights，这可能影响训练效果")
            else:
                logger.info(f"✅ 发现类别权重设置: {data_config['class_weights']}")
            
            # 检查数据路径
            for split in ['train', 'val']:
                path = data_config[split]
                if not os.path.exists(path):
                    logger.error(f"{split}数据路径不存在: {path}")
                    return False
            
            logger.info("✅ 训练设置验证通过")
            return True
            
        except Exception as e:
            logger.error(f"训练设置验证失败: {e}")
            return False


def main():
    """主函数 - 命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="训练屋顶检测模型")
    parser.add_argument("--config", default="config/model_config.yaml", help="模型配置文件")
    parser.add_argument("--data", default="config/data.yaml", help="数据配置文件")
    parser.add_argument("--project", default="runs/segment", help="项目保存路径")
    parser.add_argument("--name", default="train_improved", help="训练任务名称")
    parser.add_argument("--epochs", type=int, help="训练轮数")
    parser.add_argument("--batch", type=int, help="批次大小")
    parser.add_argument("--lr0", type=float, help="初始学习率")
    
    args = parser.parse_args()
    
    # 创建训练器
    trainer = RoofDetectionTrainer(args.config)
    
    # 验证设置
    if not trainer.validate_training_setup(args.data):
        logger.error("训练设置验证失败，请检查配置")
        return
    
    # 准备训练参数
    train_kwargs = {}
    if args.epochs:
        train_kwargs['epochs'] = args.epochs
    if args.batch:
        train_kwargs['batch'] = args.batch
    if args.lr0:
        train_kwargs['lr0'] = args.lr0
    
    # 开始训练
    try:
        results = trainer.train(
            data_yaml=args.data,
            project=args.project,
            name=args.name,
            **train_kwargs
        )
        
        # 显示最佳模型路径
        best_model = trainer.get_best_model_path(args.project, args.name)
        print(f"\n🎉 训练完成!")
        print(f"📁 最佳模型路径: {best_model}")
        print(f"📊 查看结果: {os.path.join(args.project, args.name)}")
        
    except Exception as e:
        logger.error(f"训练失败: {e}")


if __name__ == "__main__":
    main()
