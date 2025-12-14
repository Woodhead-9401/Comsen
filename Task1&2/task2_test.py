from ultralytics import YOLO
import cv2
import time
import os

# 创建保存结果的目录
os.makedirs('task2_results', exist_ok=True)

print("=" * 50)
print("任务二测试：快速视频检测演示")
print("=" * 50)

# 加载模型
model = YOLO('yolov8n.pt')

# 选项1：使用摄像头实时检测（推荐）
print("\n选项1：使用摄像头实时检测")
print("请确保摄像头已连接")
input("按回车开始摄像头检测...")

cap = cv2.VideoCapture(0)  # 0为默认摄像头

if not cap.isOpened():
    print("无法打开摄像头，尝试选项2...")
else:
    print("摄像头已打开，开始检测...")
    print("按 'q' 停止检测")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 检测
        results = model(frame, verbose=False)
        
        # 绘制结果
        annotated_frame = results[0].plot()
        
        # 显示
        cv2.imshow('YOLOv8 实时检测', annotated_frame)
        
        # 按q退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# 选项2：使用示例图片创建伪视频
print("\n选项2：使用图片创建演示视频")
print("正在创建演示视频...")

# 创建演示视频
import numpy as np

# 创建一个黑色背景
height, width = 480, 640
fps = 10
duration = 5  # 秒
total_frames = fps * duration

# 创建视频写入器
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
demo_video_path = "task2_results/demo_video.mp4"
out = cv2.VideoWriter(demo_video_path, fourcc, fps, (width, height))

# 创建一些简单的动画帧
for frame_num in range(total_frames):
    # 创建背景
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 在帧上画一些移动的形状
    x = int((frame_num / total_frames) * (width - 100))
    y = height // 2
    
    # 画一个移动的矩形
    cv2.rectangle(frame, (x, y), (x+100, y+100), (0, 255, 0), 2)
    cv2.putText(frame, "Simulated Object", (x, y-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # 写入帧
    out.write(frame)

out.release()
print(f"演示视频已创建: {demo_video_path}")

# 选项3：从网络下载示例视频
print("\n选项3：从网络下载示例视频")
print("正在下载交通视频...")

try:
    import urllib.request
    
    # 示例视频URL
    video_url = "https://ultralytics.com/videos/cars.mp4"
    video_path = "task2_results/traffic_video.mp4"
    
    urllib.request.urlretrieve(video_url, video_path)
    print(f"示例视频已下载: {video_path}")
    
    # 进行检测
    print("\n正在对交通视频进行检测...")
    cap = cv2.VideoCapture(video_path)
    
    # 只处理前100帧作为演示
    for i in range(100):
        ret, frame = cap.read()
        if not ret:
            break
        
        # 检测
        results = model(frame, verbose=False)
        
        # 绘制结果
        annotated_frame = results[0].plot()
        
        # 显示帧数
        cv2.putText(annotated_frame, f"Frame: {i}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 显示
        cv2.imshow('交通视频检测', annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
except Exception as e:
    print(f"无法下载示例视频: {e}")
    print("请使用本地视频文件进行测试")

print("\n" + "="*50)
print("任务二测试完成!")
print(f"生成的文件保存在 'task2_results/' 目录")
print("="*50)