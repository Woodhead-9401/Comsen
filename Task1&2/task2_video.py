from ultralytics import YOLO
import cv2
import time
import numpy as np
import os

# 创建保存结果的目录
os.makedirs('task2_results', exist_ok=True)

print("=" * 50)
print("任务二：YOLOv8 视频目标检测")
print("=" * 50)

# 1. 加载预训练模型
print("正在加载 YOLOv8n 模型...")
model = YOLO('yolov8n.pt')
print("模型加载成功！")

def detect_video_file(video_path, output_path=None, show=True, save=False):
    """
    对视频文件进行目标检测
    
    参数:
        video_path: 视频文件路径
        output_path: 输出视频路径
        show: 是否实时显示
        save: 是否保存结果视频
    """
    print(f"\n开始处理视频: {video_path}")
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频文件 {video_path}")
        return
    
    # 获取视频信息
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"视频信息:")
    print(f"  - 分辨率: {width}×{height}")
    print(f"  - 帧率: {fps} FPS")
    print(f"  - 总帧数: {total_frames}")
    print(f"  - 时长: {total_frames/fps:.1f} 秒")
    
    # 如果保存视频，设置视频编码器
    if save and output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 性能统计
    frame_count = 0
    total_inference_time = 0
    start_time = time.time()
    
    # 逐帧处理
    print("\n开始检测... (按 'q' 停止，按 'p' 暂停)")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # 开始计时
        inference_start = time.time()
        
        # 使用YOLO进行检测
        results = model(frame, verbose=False)
        
        # 计算推理时间
        inference_time = time.time() - inference_start
        total_inference_time += inference_time
        
        # 在帧上绘制检测结果
        annotated_frame = results[0].plot()
        
        # 添加FPS信息
        current_fps = 1.0 / inference_time if inference_time > 0 else 0
        fps_text = f"FPS: {current_fps:.1f}"
        cv2.putText(annotated_frame, fps_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 显示当前帧数
        frame_text = f"Frame: {frame_count}/{total_frames}"
        cv2.putText(annotated_frame, frame_text, (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 显示检测到的物体数量
        num_objects = len(results[0].boxes)
        object_text = f"Objects: {num_objects}"
        cv2.putText(annotated_frame, object_text, (10, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 保存结果
        if save and output_path:
            out.write(annotated_frame)
        
        # 显示结果
        if show:
            # 调整显示窗口大小
            display_height = 720
            if height > display_height:
                scale = display_height / height
                display_width = int(width * scale)
                display_frame = cv2.resize(annotated_frame, (display_width, display_height))
            else:
                display_frame = annotated_frame
            
            cv2.imshow('YOLOv8 视频检测', display_frame)
            
            # 键盘控制
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n用户停止检测")
                break
            elif key == ord('p'):
                print("\n暂停中... 按任意键继续")
                cv2.waitKey(0)
        
        # 每10帧打印一次进度
        if frame_count % 10 == 0:
            elapsed_time = time.time() - start_time
            estimated_total = (elapsed_time / frame_count) * total_frames
            remaining = estimated_total - elapsed_time
            print(f"进度: {frame_count}/{total_frames} ({frame_count/total_frames*100:.1f}%), "
                  f"剩余时间: {remaining:.1f}秒")
    
    # 计算性能统计
    elapsed_time = time.time() - start_time
    avg_fps = frame_count / elapsed_time
    avg_inference_time = total_inference_time / frame_count * 1000  # 转换为毫秒
    
    print(f"\n处理完成!")
    print(f"总时间: {elapsed_time:.1f} 秒")
    print(f"平均FPS: {avg_fps:.1f}")
    print(f"平均推理时间: {avg_inference_time:.1f} 毫秒/帧")
    
    # 释放资源
    cap.release()
    if save and output_path:
        out.release()
        print(f"结果视频已保存: {output_path}")
    
    cv2.destroyAllWindows()
    return avg_fps

def detect_camera(camera_id=0, output_path=None, duration=30):
    """
    使用摄像头进行实时检测
    
    参数:
        camera_id: 摄像头ID (0为默认摄像头)
        output_path: 输出视频路径
        duration: 检测持续时间(秒)
    """
    print(f"\n正在打开摄像头 {camera_id}...")
    
    # 打开摄像头
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"错误: 无法打开摄像头 {camera_id}")
        return
    
    # 获取摄像头参数
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print(f"摄像头信息:")
    print(f"  - 分辨率: {width}×{height}")
    print(f"  - 帧率: {fps} FPS")
    
    # 如果保存视频，设置视频编码器
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 性能统计
    frame_count = 0
    total_inference_time = 0
    start_time = time.time()
    
    print(f"\n开始实时检测... (持续 {duration} 秒，按 'q' 随时停止)")
    print("按 'q' 停止，按 'p' 暂停")
    
    while True:
        # 检查是否达到最大持续时间
        elapsed_time = time.time() - start_time
        if elapsed_time > duration:
            print(f"\n已达到最大检测时间 ({duration}秒)")
            break
        
        ret, frame = cap.read()
        if not ret:
            print("错误: 无法读取摄像头帧")
            break
        
        frame_count += 1
        
        # 开始计时
        inference_start = time.time()
        
        # 使用YOLO进行检测
        results = model(frame, verbose=False)
        
        # 计算推理时间
        inference_time = time.time() - inference_start
        total_inference_time += inference_time
        
        # 在帧上绘制检测结果
        annotated_frame = results[0].plot()
        
        # 添加性能信息
        current_fps = 1.0 / inference_time if inference_time > 0 else 0
        fps_text = f"FPS: {current_fps:.1f}"
        cv2.putText(annotated_frame, fps_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 显示已处理帧数
        frame_text = f"Frames: {frame_count}"
        cv2.putText(annotated_frame, frame_text, (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 显示检测到的物体数量
        num_objects = len(results[0].boxes)
        object_text = f"Objects: {num_objects}"
        cv2.putText(annotated_frame, object_text, (10, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 显示运行时间
        time_text = f"Time: {elapsed_time:.1f}s/{duration}s"
        cv2.putText(annotated_frame, time_text, (10, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 保存结果
        if output_path:
            out.write(annotated_frame)
        
        # 显示结果
        display_height = 720
        if height > display_height:
            scale = display_height / height
            display_width = int(width * scale)
            display_frame = cv2.resize(annotated_frame, (display_width, display_height))
        else:
            display_frame = annotated_frame
        
        cv2.imshow('YOLOv8 实时摄像头检测', display_frame)
        
        # 键盘控制
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n用户停止检测")
            break
        elif key == ord('p'):
            print("\n暂停中... 按任意键继续")
            cv2.waitKey(0)
    
    # 计算性能统计
    elapsed_time = time.time() - start_time
    avg_fps = frame_count / elapsed_time
    avg_inference_time = total_inference_time / frame_count * 1000
    
    print(f"\n检测完成!")
    print(f"总帧数: {frame_count}")
    print(f"总时间: {elapsed_time:.1f} 秒")
    print(f"平均FPS: {avg_fps:.1f}")
    print(f"平均推理时间: {avg_inference_time:.1f} 毫秒/帧")
    
    # 释放资源
    cap.release()
    if output_path:
        out.release()
        print(f"检测视频已保存: {output_path}")
    
    cv2.destroyAllWindows()
    return avg_fps

def main():
    """主函数"""
    print("请选择检测模式:")
    print("1. 视频文件检测")
    print("2. 摄像头实时检测")
    
    choice = input("请输入选项 (1 或 2): ").strip()
    
    if choice == '1':
        # 视频文件检测
        video_path = input("请输入视频文件路径 (或按回车使用示例视频): ").strip()
        
        if not video_path:
            # 如果没有提供视频文件，尝试下载一个示例视频
            print("正在下载示例视频...")
            import urllib.request
            
            # 示例视频URL
            video_url = "https://ultralytics.com/videos/cars.mp4"
            video_path = "task2_results/example_video.mp4"
            
            try:
                urllib.request.urlretrieve(video_url, video_path)
                print(f"示例视频已下载: {video_path}")
            except:
                print("无法下载示例视频，请提供本地视频文件")
                video_path = input("请输入本地视频文件路径: ").strip()
        
        output_path = "task2_results/video_detection_output.mp4"
        
        save_video = input("是否保存检测结果视频? (y/n): ").strip().lower() == 'y'
        
        detect_video_file(video_path, output_path, show=True, save=save_video)
        
    elif choice == '2':
        # 摄像头检测
        camera_id = input("请输入摄像头ID (默认0): ").strip()
        camera_id = int(camera_id) if camera_id else 0
        
        duration = input("请输入检测持续时间(秒，默认30): ").strip()
        duration = int(duration) if duration else 30
        
        save_video = input("是否保存检测视频? (y/n): ").strip().lower() == 'y'
        
        output_path = None
        if save_video:
            output_path = "task2_results/camera_detection_output.mp4"
        
        detect_camera(camera_id, output_path, duration)
    
    else:
        print("无效选项!")

if __name__ == "__main__":
    main()