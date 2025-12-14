import os
from ultralytics import YOLO
import torch

def train_coco_subset():
    # 检查GPU是否可用
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {'GPU' if device != 'cpu' else 'CPU'}")
    
    # 加载预训练模型
    print("加载YOLOv8n预训练模型...")
    model = YOLO('yolov8n.pt')  # 使用YOLOv8n（最小版本），适合快速训练
    
    # 训练参数配置
    train_args = {
        'data': 'coco_subset.yaml',    # 数据集配置文件
        'epochs': 50,                  # 训练轮数（可根据需要调整）
        'imgsz': 640,                  # 输入图像尺寸
        'batch': 16,                   # 批次大小
        'workers': 4,                  # 数据加载线程数
        'device': device,              # 使用GPU或CPU
        'project': 'coco_subset_train', # 项目名称
        'name': 'yolov8n_coco_subset', # 实验名称
        'save': True,                  # 保存模型
        'save_period': 10,             # 每10个epoch保存一次
        'pretrained': True,            # 使用预训练权重
        'optimizer': 'auto',           # 自动选择优化器
        'lr0': 0.01,                   # 初始学习率
        'lrf': 0.01,                   # 最终学习率因子
        'momentum': 0.937,             # 动量
        'weight_decay': 0.0005,        # 权重衰减
        'warmup_epochs': 3.0,          # 预热epoch数
        'warmup_momentum': 0.8,        # 预热动量
        'box': 7.5,                    # 边界框损失权重
        'cls': 0.5,                    # 分类损失权重
        'dfl': 1.5,                    # DFL损失权重
        'hsv_h': 0.015,                # 图像HSV-Hue增强
        'hsv_s': 0.7,                  # 图像HSV-Saturation增强
        'hsv_v': 0.4,                  # 图像HSV-Value增强
        'degrees': 0.0,                # 图像旋转
        'translate': 0.1,              # 图像平移
        'scale': 0.5,                  # 图像缩放
        'shear': 0.0,                  # 图像剪切
        'perspective': 0.0,            # 图像透视变换
        'flipud': 0.0,                 # 上下翻转概率
        'fliplr': 0.5,                 # 左右翻转概率
        'mosaic': 1.0,                 # Mosaic数据增强概率
        'mixup': 0.0,                  # MixUp数据增强概率
        'copy_paste': 0.0,             # 复制粘贴数据增强概率
    }
    
    # 开始训练
    print("开始训练...")
    results = model.train(**train_args)
    
    print("训练完成!")
    return results

if __name__ == "__main__":
    # 设置环境变量（可选）
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    # 开始训练
    train_coco_subset()