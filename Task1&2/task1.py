from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import os

# 创建保存结果的目录
os.makedirs('task1_results', exist_ok=True)

print("=" * 50)
print("任务一：使用YOLOv8预训练模型进行图像检测")
print("=" * 50)

# 1. 加载预训练模型
print("正在加载YOLOv8n模型...")
model = YOLO('yolov8n.pt')  # 会自动使用已下载的模型
print("模型加载成功！")

# 2. 准备测试图片（使用在线图片或本地图片）
test_images = [
    'https://ultralytics.com/images/bus.jpg',  # Ultralytics提供的测试图片
    'https://ultralytics.com/images/zidane.jpg',
    'https://ultralytics.com/images/bus.jpg'  # 重复使用，你也可以换成其他图片
]

# 或者使用本地图片（如果有的话）
# test_images = ['image1.jpg', 'image2.jpg', 'image3.jpg']

# 3. 对每张图片进行检测
for i, img_path in enumerate(test_images):
    print(f"\n{'='*30}")
    print(f"检测图片 {i+1}: {img_path}")
    print('='*30)
    
    # 执行检测
    results = model(img_path)
    
    # 保存结果图片
    result_img_path = f'task1_results/detection_result_{i+1}.jpg'
    results[0].save(filename=result_img_path)
    print(f"✓ 结果已保存到: {result_img_path}")
    
    # 显示检测统计信息
    result = results[0]
    print(f"检测到 {len(result.boxes)} 个对象")
    
    if len(result.boxes) > 0:
        print("检测到的物体:")
        for j, box in enumerate(result.boxes):
            class_id = int(box.cls)
            class_name = result.names[class_id]
            confidence = float(box.conf)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # 获取坐标中心点
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            print(f"  {j+1}. {class_name}")
            print(f"     置信度: {confidence:.3f}")
            print(f"     边界框: [{x1}, {y1}, {x2}, {y2}]")
            print(f"     中心点: ({center_x}, {center_y})")
            print(f"     宽高: {x2-x1} × {y2-y1}")
    
    # 在控制台打印检测结果（ASCII格式）
    print("\n检测结果预览:")
    print(result.plot(show_conf=True))

# 4. 可选：使用OpenCV显示结果图片
print("\n" + "="*50)
print("使用OpenCV显示检测结果...")
print("按任意键切换到下一张图片，按'q'退出")

for i in range(len(test_images)):
    result_path = f'task1_results/detection_result_{i+1}.jpg'
    
    if os.path.exists(result_path):
        img = cv2.imread(result_path)
        
        # 调整图片大小以便显示
        height, width = img.shape[:2]
        if width > 1200:
            scale = 1200 / width
            new_width = 1200
            new_height = int(height * scale)
            img = cv2.resize(img, (new_width, new_height))
        
        cv2.imshow(f'Result {i+1}', img)
        
        # 等待按键
        key = cv2.waitKey(0)
        if key == ord('q'):
            break
        cv2.destroyAllWindows()

print("\n" + "="*50)
print("任务一完成！")
print(f"所有结果已保存到 'task1_results/' 目录")
print("="*50)