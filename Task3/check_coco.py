import os
import json
from PIL import Image

def check_coco_dataset(coco_dir):
    """检查COCO数据集是否完整"""
    print("检查COCO数据集...")
    
    # 检查必要目录
    required_dirs = ['annotations', 'train2017', 'val2017']
    for dir_name in required_dirs:
        dir_path = os.path.join(coco_dir, dir_name)
        if not os.path.exists(dir_path):
            print(f"❌ 缺少目录: {dir_path}")
            return False
        print(f"✅ 目录存在: {dir_path}")
    
    # 检查标注文件
    annotation_files = ['instances_train2017.json', 'instances_val2017.json']
    for ann_file in annotation_files:
        ann_path = os.path.join(coco_dir, 'annotations', ann_file)
        if not os.path.exists(ann_path):
            print(f"❌ 缺少标注文件: {ann_path}")
            return False
        print(f"✅ 标注文件存在: {ann_path}")
        
        # 检查JSON是否有效
        try:
            with open(ann_path, 'r') as f:
                data = json.load(f)
            print(f"  - 包含 {len(data.get('images', []))} 张图像")
            print(f"  - 包含 {len(data.get('annotations', []))} 个标注")
            print(f"  - 包含 {len(data.get('categories', []))} 个类别")
        except Exception as e:
            print(f"❌ 标注文件损坏: {e}")
            return False
    
    # 检查图像数量
    train_dir = os.path.join(coco_dir, 'train2017')
    val_dir = os.path.join(coco_dir, 'val2017')
    
    train_images = [f for f in os.listdir(train_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    val_images = [f for f in os.listdir(val_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    print(f"✅ 训练集图像: {len(train_images)} 张")
    print(f"✅ 验证集图像: {len(val_images)} 张")
    
    # 检查几张样本图像
    print("\n检查样本图像...")
    sample_images = train_images[:3] if train_images else []
    for img_name in sample_images:
        img_path = os.path.join(train_dir, img_name)
        try:
            with Image.open(img_path) as img:
                print(f"✅ 图像有效: {img_name} ({img.size[0]}x{img.size[1]})")
        except Exception as e:
            print(f"❌ 图像损坏: {img_name} - {e}")
            return False
    
    print("\n✅ COCO数据集检查完成，一切正常！")
    return True

if __name__ == "__main__":
    # 修改为你的COCO数据集路径
    COCO_DIR = "D:/coco2017"  # 修改这里！
    
    if check_coco_dataset(COCO_DIR):
        print("\n可以开始准备数据子集了！")
        print("运行: python prepare_coco_subset.py")
    else:
        print("\n数据集有问题，请检查路径和文件！")