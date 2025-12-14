import json
import os
import shutil
from tqdm import tqdm
import random

def prepare_coco_subset(coco_dir, output_dir, target_classes, train_limit=1000, val_limit=200):
    """
    从COCO数据集中提取指定类别的子集
    
    Args:
        coco_dir: COCO数据集目录
        output_dir: 输出目录
        target_classes: 目标类别ID列表
        train_limit: 训练集图像数量限制
        val_limit: 验证集图像数量限制
    """
    
    # 创建输出目录结构
    os.makedirs(os.path.join(output_dir, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', 'val'), exist_ok=True)
    
    # 类别ID到名称映射
    coco_categories = {
        1: 'person',
        2: 'bicycle',
        3: 'car',
        4: 'motorcycle',
        5: 'airplane',
        6: 'bus',
        7: 'train',
        8: 'truck',
        9: 'boat',
        10: 'traffic light',
        11: 'fire hydrant',
        13: 'stop sign',
        14: 'parking meter',
        15: 'bench',
        16: 'bird',
        17: 'cat',
        18: 'dog',
        19: 'horse',
        20: 'sheep',
        21: 'cow',
        22: 'elephant',
        23: 'bear',
        24: 'zebra',
        25: 'giraffe',
        27: 'backpack',
        28: 'umbrella',
        31: 'handbag',
        32: 'tie',
        33: 'suitcase',
        34: 'frisbee',
        35: 'skis',
        36: 'snowboard',
        37: 'sports ball',
        38: 'kite',
        39: 'baseball bat',
        40: 'baseball glove',
        41: 'skateboard',
        42: 'surfboard',
        43: 'tennis racket',
        44: 'bottle',
        46: 'wine glass',
        47: 'cup',
        48: 'fork',
        49: 'knife',
        50: 'spoon',
        51: 'bowl',
        52: 'banana',
        53: 'apple',
        54: 'sandwich',
        55: 'orange',
        56: 'broccoli',
        57: 'carrot',
        58: 'hot dog',
        59: 'pizza',
        60: 'donut',
        61: 'cake',
        62: 'chair',
        63: 'couch',
        64: 'potted plant',
        65: 'bed',
        67: 'dining table',
        70: 'toilet',
        72: 'tv',
        73: 'laptop',
        74: 'mouse',
        75: 'remote',
        76: 'keyboard',
        77: 'cell phone',
        78: 'microwave',
        79: 'oven',
        80: 'toaster',
        81: 'sink',
        82: 'refrigerator',
        83: 'book',
        84: 'clock',
        85: 'vase',
        86: 'scissors',
        87: 'teddy bear',
        88: 'hair drier',
        89: 'toothbrush'
    }
    
    # 目标类别ID到新ID的映射
    target_class_mapping = {class_id: idx for idx, class_id in enumerate(target_classes)}
    
    # 处理训练集
    print("处理训练集...")
    with open(os.path.join(coco_dir, 'annotations/instances_train2017.json'), 'r') as f:
        train_ann = json.load(f)
    
    # 筛选包含目标类别的图像
    train_images_with_target = set()
    for ann in train_ann['annotations']:
        if ann['category_id'] in target_classes:
            train_images_with_target.add(ann['image_id'])
    
    # 限制图像数量
    train_images_with_target = list(train_images_with_target)[:train_limit]
    
    # 图像ID到文件名的映射
    image_id_to_file = {img['id']: img['file_name'] for img in train_ann['images']}
    
    # 为每个图像创建YOLO格式标注
    for img_id in tqdm(train_images_with_target):
        if img_id not in image_id_to_file:
            continue
            
        img_file = image_id_to_file[img_id]
        img_path = os.path.join(coco_dir, 'train2017', img_file)
        
        # 复制图像
        shutil.copy(img_path, os.path.join(output_dir, 'images', 'train', img_file))
        
        # 创建标注文件
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(output_dir, 'labels', 'train', label_file)
        
        with open(label_path, 'w') as f:
            # 查找该图像的所有标注
            for ann in train_ann['annotations']:
                if ann['image_id'] == img_id and ann['category_id'] in target_classes:
                    # 获取COCO格式的边界框 [x_min, y_min, width, height]
                    bbox = ann['bbox']
                    
                    # 获取图像尺寸
                    img_info = next(img for img in train_ann['images'] if img['id'] == img_id)
                    img_width, img_height = img_info['width'], img_info['height']
                    
                    # 转换为YOLO格式 [x_center, y_center, width, height] (归一化)
                    x_center = (bbox[0] + bbox[2] / 2) / img_width
                    y_center = (bbox[1] + bbox[3] / 2) / img_height
                    width = bbox[2] / img_width
                    height = bbox[3] / img_height
                    
                    # 获取新的类别ID
                    new_class_id = target_class_mapping[ann['category_id']]
                    
                    # 写入文件
                    f.write(f"{new_class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    # 处理验证集
    print("\n处理验证集...")
    with open(os.path.join(coco_dir, 'annotations/instances_val2017.json'), 'r') as f:
        val_ann = json.load(f)
    
    # 筛选包含目标类别的图像
    val_images_with_target = set()
    for ann in val_ann['annotations']:
        if ann['category_id'] in target_classes:
            val_images_with_target.add(ann['image_id'])
    
    # 限制图像数量
    val_images_with_target = list(val_images_with_target)[:val_limit]
    
    # 图像ID到文件名的映射
    val_image_id_to_file = {img['id']: img['file_name'] for img in val_ann['images']}
    
    # 为每个图像创建YOLO格式标注
    for img_id in tqdm(val_images_with_target):
        if img_id not in val_image_id_to_file:
            continue
            
        img_file = val_image_id_to_file[img_id]
        img_path = os.path.join(coco_dir, 'val2017', img_file)
        
        # 复制图像
        shutil.copy(img_path, os.path.join(output_dir, 'images', 'val', img_file))
        
        # 创建标注文件
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(output_dir, 'labels', 'val', label_file)
        
        with open(label_path, 'w') as f:
            # 查找该图像的所有标注
            for ann in val_ann['annotations']:
                if ann['image_id'] == img_id and ann['category_id'] in target_classes:
                    # 获取COCO格式的边界框
                    bbox = ann['bbox']
                    
                    # 获取图像尺寸
                    img_info = next(img for img in val_ann['images'] if img['id'] == img_id)
                    img_width, img_height = img_info['width'], img_info['height']
                    
                    # 转换为YOLO格式
                    x_center = (bbox[0] + bbox[2] / 2) / img_width
                    y_center = (bbox[1] + bbox[3] / 2) / img_height
                    width = bbox[2] / img_width
                    height = bbox[3] / img_height
                    
                    # 获取新的类别ID
                    new_class_id = target_class_mapping[ann['category_id']]
                    
                    # 写入文件
                    f.write(f"{new_class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    print(f"\n数据集准备完成！")
    print(f"训练集图像数量: {len(train_images_with_target)}")
    print(f"验证集图像数量: {len(val_images_with_target)}")
    
    # 创建类别名称映射
    class_names = {target_class_mapping[k]: coco_categories[k] for k in target_classes}
    print(f"\n类别映射: {class_names}")
    
    return class_names

if __name__ == "__main__":
    # 配置参数
    COCO_DIR = "D:/coco2017"  # 修改为你的COCO数据集路径
    OUTPUT_DIR = "./coco_subset_3class"
    TARGET_CLASSES = [1, 3, 18]  # person, car, dog
    
    # 准备数据集
    class_names = prepare_coco_subset(
        coco_dir=COCO_DIR,
        output_dir=OUTPUT_DIR,
        target_classes=TARGET_CLASSES,
        train_limit=1000,  # 训练集图像数量
        val_limit=200      # 验证集图像数量
    )