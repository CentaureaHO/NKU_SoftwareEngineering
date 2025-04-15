import mediapipe as mp
import cv2
import numpy as np

# 参数初始值
max_hands = 2
detection_confidence = 50  # 0-100之间的值，会转换为0.0-1.0
tracking_confidence = 50   # 0-100之间的值，会转换为0.0-1.0
landmark_color_r = 0       # 0-255
landmark_color_g = 0       # 0-255
landmark_color_b = 255     # 0-255
box_color_r = 0            # 0-255
box_color_g = 255          # 0-255
box_color_b = 0            # 0-255
box_thickness = 2          # 1-10
landmark_size = 5          # 1-10

# 创建控制窗口
cv2.namedWindow('Controls')
cv2.resizeWindow('Controls', 600, 400)

# 创建滑块
def nothing(x):
    pass

cv2.createTrackbar('Max Hands', 'Controls', max_hands, 4, nothing)
cv2.createTrackbar('Detection Conf %', 'Controls', detection_confidence, 100, nothing)
cv2.createTrackbar('Tracking Conf %', 'Controls', tracking_confidence, 100, nothing)
cv2.createTrackbar('Landmark R', 'Controls', landmark_color_r, 255, nothing)
cv2.createTrackbar('Landmark G', 'Controls', landmark_color_g, 255, nothing)
cv2.createTrackbar('Landmark B', 'Controls', landmark_color_b, 255, nothing)
cv2.createTrackbar('Box R', 'Controls', box_color_r, 255, nothing)
cv2.createTrackbar('Box G', 'Controls', box_color_g, 255, nothing)
cv2.createTrackbar('Box B', 'Controls', box_color_b, 255, nothing)
cv2.createTrackbar('Box Thickness', 'Controls', box_thickness, 10, nothing)
cv2.createTrackbar('Landmark Size', 'Controls', landmark_size, 10, nothing)

# 初始化MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=max_hands,
    min_detection_confidence=detection_confidence/100,
    min_tracking_confidence=tracking_confidence/100
)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# 初始化摄像头
cap = cv2.VideoCapture(0)

# 用于绘制边框的函数
def draw_hand_box(image, landmarks, color, thickness):
    x_min, y_min = min([landmark.x for landmark in landmarks]), min([landmark.y for landmark in landmarks])
    x_max, y_max = max([landmark.x for landmark in landmarks]), max([landmark.y for landmark in landmarks])
    
    h, w, _ = image.shape
    x_min, y_min = int(x_min * w), int(y_min * h)
    x_max, y_max = int(x_max * w), int(y_max * h)
    
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)

# 读取参数并更新Hands模型
def update_hands_model():
    global hands, max_hands, detection_confidence, tracking_confidence
    
    new_max_hands = cv2.getTrackbarPos('Max Hands', 'Controls')
    new_detection_confidence = cv2.getTrackbarPos('Detection Conf %', 'Controls') / 100
    new_tracking_confidence = cv2.getTrackbarPos('Tracking Conf %', 'Controls') / 100
    
    # 只有当参数变化时才重新初始化
    if (new_max_hands != max_hands or 
        new_detection_confidence != detection_confidence/100 or 
        new_tracking_confidence != tracking_confidence/100):
        
        max_hands = new_max_hands
        detection_confidence = int(new_detection_confidence * 100)
        tracking_confidence = int(new_tracking_confidence * 100)
        
        # 释放旧模型并创建新模型
        hands.close()
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=new_detection_confidence,
            min_tracking_confidence=new_tracking_confidence
        )

def get_landmark_drawing_spec():
    r = cv2.getTrackbarPos('Landmark R', 'Controls')
    g = cv2.getTrackbarPos('Landmark G', 'Controls')
    b = cv2.getTrackbarPos('Landmark B', 'Controls')
    size = cv2.getTrackbarPos('Landmark Size', 'Controls')
    return mp_drawing.DrawingSpec(color=(r, g, b), thickness=size, circle_radius=size)

def get_box_color_and_thickness():
    r = cv2.getTrackbarPos('Box R', 'Controls')
    g = cv2.getTrackbarPos('Box G', 'Controls')
    b = cv2.getTrackbarPos('Box B', 'Controls')
    thickness = cv2.getTrackbarPos('Box Thickness', 'Controls')
    if thickness == 0:  # 避免线宽为0
        thickness = 1
    return (b, g, r), thickness  # OpenCV用BGR格式

last_update_time = 0
fps_counter = 0
fps = 0
start_time = cv2.getTickCount()

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("摄像头读取失败")
        break
    
    # 更新FPS
    current_time = cv2.getTickCount()
    if current_time - start_time > cv2.getTickFrequency():
        fps = fps_counter
        fps_counter = 0
        start_time = current_time
        
        # 每秒更新一次模型参数
        update_hands_model()
    fps_counter += 1
    
    # 水平翻转图像，使其如镜像
    image = cv2.flip(image, 1)
    
    # 将BGR图像转换为RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 处理图像
    results = hands.process(image_rgb)
    
    # 获取当前绘图设置
    landmark_spec = get_landmark_drawing_spec()
    box_color, box_thickness = get_box_color_and_thickness()
    
    # 在图像上绘制手部关键点和边框
    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            # 获取手的类型（左手或右手）
            hand_label = handedness.classification[0].label  # "Left" 或 "Right"
            
            # 绘制21个关键点和连接线
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                landmark_spec,
                mp_drawing.DrawingSpec(color=landmark_spec.color, thickness=landmark_spec.thickness)
            )
            
            # 绘制手部边框
            draw_hand_box(image, hand_landmarks.landmark, box_color, box_thickness)
            
            # 显示手的类型
            h, w, _ = image.shape
            x_min = min([landmark.x for landmark in hand_landmarks.landmark])
            y_min = min([landmark.y for landmark in hand_landmarks.landmark])
            x_min, y_min = int(x_min * w), int(y_min * h)
            cv2.putText(image, hand_label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    # 显示FPS
    fps_text = f"FPS: {fps}"
    cv2.putText(image, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # 显示图像
    cv2.imshow('MediaPipe Hands', image)
    
    # 按ESC键退出
    if cv2.waitKey(5) & 0xFF == 27:
        break

# 释放资源
hands.close()
cap.release()
cv2.destroyAllWindows()