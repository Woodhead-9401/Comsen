from ultralytics import YOLO
import matplotlib.pyplot as plt
import os

def evaluate_model():
    # 加载训练好的最佳模型
    model_path = 'coco_subset_train/yolov8n_coco_subset2/weights/best.pt'
    
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        print("请先训练模型或检查路径")
        return
    
    print(f"加载模型: {model_path}")
    model = YOLO(model_path)
    
    # 验证模型
    print("评估模型性能...")
    metrics = model.val(
        data='coco_subset.yaml',
        split='val',
        imgsz=640,
        batch=16,
        conf=0.25,  # 置信度阈值
        iou=0.45,   # IoU阈值
        device='cpu'  # 可以使用'cuda'如果GPU可用
    )
    
    # 打印评估结果
    print("\n" + "="*50)
    print("模型评估结果:")
    print("="*50)
    print(f"mAP50 (IoU=0.5): {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    
    # 按类别显示AP
    if hasattr(metrics.box, 'ap_class_index'):
        print("\n各个类别的AP50:")
        for i, class_idx in enumerate(metrics.box.ap_class_index):
            class_name = model.names[class_idx] if hasattr(model, 'names') else f'Class {class_idx}'
            ap50 = metrics.box.ap50[i]
            print(f"  {class_name}: {ap50:.4f}")
    
    # 显示混淆矩阵（如果生成了）
    confusion_matrix_path = 'coco_subset_train/yolov8n_coco_subset/confusion_matrix.png'
    if os.path.exists(confusion_matrix_path):
        print(f"\n混淆矩阵已保存至: {confusion_matrix_path}")
    
    # 显示F1-置信度曲线
    f1_curve_path = 'coco_subset_train/yolov8n_coco_subset/F1_curve.png'
    if os.path.exists(f1_curve_path):
        print(f"F1-置信度曲线已保存至: {f1_curve_path}")
    
    # 显示PR曲线
    pr_curve_path = 'coco_subset_train/yolov8n_coco_subset/PR_curve.png'
    if os.path.exists(pr_curve_path):
        print(f"PR曲线已保存至: {pr_curve_path}")
    
    return metrics

if __name__ == "__main__":
    evaluate_model()