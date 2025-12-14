from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import os
import warnings

# 忽略matplotlib的字体警告
warnings.filterwarnings('ignore', category=UserWarning)

# 设置matplotlib使用支持中文的字体（Windows系统）
import matplotlib
try:
    # 尝试使用微软雅黑字体
    matplotlib.font_manager.fontManager.addfont("C:\\Windows\\Fonts\\msyh.ttc")
    matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    matplotlib.rcParams['axes.unicode_minus'] = False
except:
    # 如果失败，使用默认英文字体，避免中文字符
    pass

# 创建保存结果的目录
os.makedirs('task1_results', exist_ok=True)

print("=" * 50)
print("Task 1: Image Detection with YOLOv8 Pretrained Model")
print("=" * 50)

# 1. Load pretrained model
print("Loading YOLOv8n model...")
model = YOLO('yolov8n.pt')
print("Model loaded successfully!")

# 2. Prepare test images
test_images = [
    'https://ultralytics.com/images/bus.jpg',
    'https://ultralytics.com/images/zidane.jpg',
    'https://ultralytics.com/images/bus.jpg'
]

# 3. Detect each image
for i, img_path in enumerate(test_images):
    print(f"\n{'='*40}")
    print(f"Processing Image {i+1}: {img_path}")
    print('='*40)
    
    # Perform detection
    results = model(img_path)
    
    # Save result image
    result_img_path = f'task1_results/detection_result_{i+1}.jpg'
    results[0].save(filename=result_img_path)
    print(f"✓ Result saved to: {result_img_path}")
    
    # Display detection statistics
    result = results[0]
    print(f"Detected {len(result.boxes)} objects")
    
    if len(result.boxes) > 0:
        print("Detected objects:")
        for j, box in enumerate(result.boxes):
            class_id = int(box.cls)
            class_name = result.names[class_id]
            confidence = float(box.conf)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            print(f"  {j+1}. {class_name}")
            print(f"     Confidence: {confidence:.3f}")
            print(f"     Bounding Box: [{x1}, {y1}, {x2}, {y2}]")
            print(f"     Center: ({center_x}, {center_y})")
            print(f"     Dimensions: {x2-x1} × {y2-y1}")
    
    print("\nDetection completed successfully!")

# 4. Create summary report without Chinese characters
print("\n" + "="*50)
print("Task 1 Summary Report")
print("="*50)

# Collect all detection results
all_detections = []
for i in range(len(test_images)):
    result_img_path = f'task1_results/detection_result_{i+1}.jpg'
    if os.path.exists(result_img_path):
        # Read and display result
        img = cv2.imread(result_img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Create subplot (use English titles only)
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, i+1)
        plt.imshow(img_rgb)
        plt.title(f'Result {i+1}')
        plt.axis('off')
        
        # Statistics
        all_detections.append({
            'image': f'Result{i+1}',
            'path': result_img_path,
            'exists': os.path.exists(result_img_path),
            'size': os.path.getsize(result_img_path)
        })

plt.tight_layout()
plt.savefig('task1_results/summary_report.jpg', dpi=150, bbox_inches='tight')
plt.show()

print("Detection results summary:")
for det in all_detections:
    if det['exists']:
        size_kb = det['size'] / 1024
        print(f"  {det['image']}: {det['path']} ({size_kb:.1f} KB)")
    else:
        print(f"  {det['image']}: File not generated")

# 5. Display individual results with OpenCV
print("\nDisplaying detection results...")
print("Press any key to show next image, 'q' to quit")

for i in range(min(3, len(test_images))):
    result_path = f'task1_results/detection_result_{i+1}.jpg'
    
    if os.path.exists(result_path):
        img = cv2.imread(result_path)
        height, width = img.shape[:2]
        
        # Resize if too large
        max_width = 1200
        if width > max_width:
            scale = max_width / width
            new_width = max_width
            new_height = int(height * scale)
            img = cv2.resize(img, (new_width, new_height))
        
        cv2.imshow(f'YOLOv8 Detection Result {i+1}', img)
        
        # Wait for key press
        key = cv2.waitKey(0)
        if key == ord('q') or key == 27:  # 'q' or ESC
            break
        cv2.destroyAllWindows()
    else:
        print(f"Warning: {result_path} does not exist")

cv2.destroyAllWindows()

print("\n" + "="*50)
print("Task 1 Completed Successfully!")
print("="*50)
print("Generated files:")
for file in os.listdir('task1_results'):
    file_path = os.path.join('task1_results', file)
    if os.path.isfile(file_path):
        size = os.path.getsize(file_path) / 1024
        print(f"  - {file}: {size:.1f} KB")
print("="*50)