import mediapipe as mp
import cv2
import csv
import numpy as np
import os
import time

# 初始化参数
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
cap = cv2.VideoCapture(0)

# 检查是否已存在数据文件，只在不存在时创建新文件
file_path = 'Dataset/gesture_data.csv'
file_exists = os.path.isfile(file_path)
if not file_exists:
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = [f'{axis}{i}' for i in range(21) for axis in ['x', 'y', 'z']] + ['label']
        writer.writerow(header)
    print("创建了新的数据文件:", file_path)
else:
    print("使用现有数据文件:", file_path)

print("""
数据采集说明：
1. 将手放在摄像头前保持手势
2. 按下数字键0-9开始录制对应手势
3. 每个手势会自动连续采集多个样本
4. 按空格键停止录制
5. 按ESC键退出程序
""")

# 添加全局变量
is_recording = False
recording_label = None
frames_count = 0
recording_interval = 3  # 每隔多少帧保存一次
max_samples = 1000  # 每次录制的帧数
samples_saved = 0
last_save_time = time.time()

def check_hand_quality(landmarks):
    """检查手部关键点质量"""
    # 检查手腕位置是否在合理范围内
    wrist = landmarks[0]
    if not (0.1 < wrist.x < 0.9 and 0.1 < wrist.y < 0.9):
        return False
    
    # 检查手指关键点是否都在视野内
    for lm in landmarks:
        if not (0.05 < lm.x < 0.95 and 0.05 < lm.y < 0.95):
            return False
    
    # 检查手指是否展开（避免手指重叠）
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]
    
    # 计算指尖之间的距离
    tips = [thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip]
    for i in range(len(tips)):
        for j in range(i+1, len(tips)):
            dist = np.sqrt((tips[i].x - tips[j].x)**2 + (tips[i].y - tips[j].y)**2)
            if dist < 0.05:  # 如果指尖距离太近
                return False
    
    return True

def calculate_hand_size(landmarks):
    """计算手部大小"""
    x_coords = [lm.x for lm in landmarks]
    y_coords = [lm.y for lm in landmarks]
    width = max(x_coords) - min(x_coords)
    height = max(y_coords) - min(y_coords)
    return width * height

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue
    
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    # 显示手部检测状态
    hand_detected = results.multi_hand_landmarks is not None
    cv2.putText(image, f"Hand detected: {hand_detected}", 
               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    # 显示实时画面和采集状态
    if is_recording:
        cv2.putText(image, f"Recording gesture {recording_label}: {samples_saved}/{max_samples//recording_interval}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        cv2.putText(image, "Press 0-9 to start recording, ESC to exit", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # 绘制手部关键点以便更好地定位
    if results.multi_hand_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            image, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
    
    cv2.imshow('Data Collection', image)
    
    # 连续采集逻辑
    if is_recording and results.multi_hand_landmarks:
        current_time = time.time()
        if current_time - last_save_time >= 0.1:  # 限制保存频率
            landmarks = results.multi_hand_landmarks[0].landmark
            
            # 检查手部质量
            if check_hand_quality(landmarks):
                # 计算手部大小
                hand_size = calculate_hand_size(landmarks)
                if 0.1 < hand_size < 0.5:  # 手部大小在合理范围内
                    row = [lm.x for lm in landmarks] + [lm.y for lm in landmarks] + [lm.z for lm in landmarks] + [recording_label]
                    try:
                        with open(file_path, 'a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow(row)
                        samples_saved += 1
                        print(f"成功保存样本 {samples_saved} 到 {file_path}")
                        last_save_time = current_time
                    except Exception as e:
                        print(f"保存数据出错: {e}")
                else:
                    print("手部大小不合适，未保存")
            else:
                print("手部质量检查未通过，未保存")
        
        if samples_saved >= max_samples//recording_interval:
            print(f"完成手势 {recording_label} 的采集，共 {samples_saved} 个样本")
            is_recording = False
            samples_saved = 0
    
    key = cv2.waitKey(5)
    if key == 27:  # ESC
        break
    elif 48 <= key <= 57:  # 数字键0-9
        recording_label = key - 48
        is_recording = True
        samples_saved = 0
        last_save_time = time.time()
        print(f"开始录制手势 {recording_label}")
    elif key == 32:  # 空格键停止录制
        is_recording = False
        print("停止录制")

hands.close()
cap.release()
cv2.destroyAllWindows()

print(f"数据已保存到 {os.path.abspath(file_path)}")