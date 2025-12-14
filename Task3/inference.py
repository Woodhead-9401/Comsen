from ultralytics import YOLO
import cv2
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def run_inference():
    # 加载训练好的模型
    model_path = 'coco_subset_train/yolov8n_coco_subset2y/weights/best.pt'
    
    # 检查模型是否存在
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        print("请先训练模型!")
        return None
    
    model = YOLO(model_path)
    
    # 测试图像路径 - 使用验证集中的图像
    test_images = []
    val_dir = 'coco_subset_3class/images/val'
    
    if os.path.exists(val_dir):
        # 获取验证集的前3张图像
        val_images = [f for f in os.listdir(val_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        test_images = [os.path.join(val_dir, img) for img in val_images[:3]]
    else:
        print(f"验证集目录不存在: {val_dir}")
        print("使用备用测试图像...")
        # 备用：使用示例图像
        test_images = [
            'https://ultralytics.com/images/bus.jpg',
            'https://ultralytics.com/images/zidane.jpg'
        ]
    
    results = []
    for img_path in test_images:
        print(f"\n处理图像: {img_path}")
        
        try:
            # 进行推理
            result = model(img_path, conf=0.5)  # 置信度阈值0.5
            
            # 显示结果
            plotted_img = result[0].plot()  # 绘制检测结果
            
            # 保存结果图像
            if img_path.startswith('http'):
                output_path = f"detected_{img_path.split('/')[-1]}"
            else:
                output_path = img_path.replace('.jpg', '_detected.jpg')
            
            cv2.imwrite(output_path, plotted_img)
            print(f"结果保存至: {output_path}")
            
            # 打印检测到的对象
            if len(result[0].boxes) > 0:
                print(f"检测到 {len(result[0].boxes)} 个对象:")
                for i, box in enumerate(result[0].boxes):
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    confidence = float(box.conf[0])
                    print(f"  {i+1}. {class_name}: {confidence:.2f}")
            else:
                print("未检测到任何对象")
            
            results.append(result)
            
        except Exception as e:
            print(f"处理图像 {img_path} 时出错: {e}")
    
    return results

def test_on_video():
    """在视频上进行测试"""
    model_path = 'coco_subset_train/yolov8n_coco_subset2/weights/best.pt'
    
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        print("请先训练模型!")
        return
    
    model = YOLO(model_path)
    
    # 使用摄像头
    cap = cv2.VideoCapture(0)  # 0表示默认摄像头
    
    if not cap.isOpened():
        print("无法打开摄像头，尝试使用测试视频...")
        # 如果没有摄像头，使用测试视频
        test_video = 'https://ultralytics.com/images/bus.jpg'
        cap = cv2.VideoCapture(test_video)
    
    if not cap.isOpened():
        print("无法打开视频源")
        return
    
    print("按'q'键退出视频检测")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        # 每隔5帧处理一次，减少计算量
        if frame_count % 5 != 0:
            continue
        
        # 调整帧大小
        frame = cv2.resize(frame, (640, 640))
        
        # 进行推理
        results = model(frame, conf=0.5)
        
        # 绘制检测结果
        annotated_frame = results[0].plot()
        
        # 显示FPS
        cv2.putText(annotated_frame, f"FPS: {int(cap.get(cv2.CAP_PROP_FPS))}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 显示结果
        cv2.imshow('YOLOv8 COCO Subset Detection', annotated_frame)
        
        # 按'q'退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("="*50)
    print("COCO子集训练模型推理测试")
    print("="*50)
    
    # 在图像上测试
    print("\n1. 在测试图像上运行推理:")
    results = run_inference()
    
    # 在视频/摄像头流上测试（可选）
    print("\n2. 在摄像头视频流上测试 (可选):")
    use_camera = input("是否使用摄像头进行测试? (y/n): ").lower()
    if use_camera == 'y':
        test_on_video()