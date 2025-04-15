import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time
import os
from collections import deque
from PIL import Image, ImageDraw, ImageFont
from simplified_preprocessing import extract_realtime_features

# 创建目录
os.makedirs('model_data', exist_ok=True)

# 初始化MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# 全局变量
MODEL_PATH = 'model_data/gesture_model.h5'
FEATURE_MEAN_PATH = 'model_data/feature_mean.npy'
FEATURE_SCALE_PATH = 'model_data/feature_scale.npy'

# 手势名称
GESTURE_NAMES = {
    0: "0", 1: "1", 2: "2", 3: "3", 4: "4",
    5: "5", 6: "6", 7: "7", 8: "8", 9: "9"
}

# 中文字体支持
def put_chinese_text(img, text, position, font_size=36, color=(0, 0, 255)):
    """在图片上添加中文文本"""
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    try:
        font = ImageFont.truetype("simhei.ttf", font_size)
    except IOError:
        try:
            font = ImageFont.truetype("simsun.ttc", font_size)
        except IOError:
            font = ImageFont.load_default()
    
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

class GestureRecognizer:
    def __init__(self):
        # 初始化状态
        self.model = None
        self.feature_mean = None
        self.feature_scale = None
        self.loaded = False
        
        # 手势历史记录
        self.gesture_history = deque(maxlen=7)
        self.confidence_history = deque(maxlen=7)
        
        # 置信度阈值
        self.confidence_threshold = 0.65
        
        # 初始化MediaPipe
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # 加载资源
        self.load_resources()
    
    def load_resources(self):
        """加载模型和标准化参数"""
        try:
            # 检查模型文件
            if not os.path.exists(MODEL_PATH):
                print(f"错误: 模型文件不存在 {MODEL_PATH}")
                return False
                
            # 加载模型
            self.model = tf.keras.models.load_model(MODEL_PATH)
            print("模型加载成功!")
            
            # 加载标准化参数
            self.feature_mean = np.load(FEATURE_MEAN_PATH)
            self.feature_scale = np.load(FEATURE_SCALE_PATH)
            print("标准化参数加载成功!")
            
            self.loaded = True
            return True
        
        except Exception as e:
            print(f"资源加载失败: {e}")
            # 退回到备用模型
            self.try_load_backup_model()
            return self.loaded
    
    def try_load_backup_model(self):
        """尝试加载备用模型"""
        try:
            backup_paths = [
                'best_model.h5',
                'final_gesture_model.h5',
                'best_gesture_model.h5'
            ]
            
            for path in backup_paths:
                if os.path.exists(path):
                    print(f"尝试加载备用模型: {path}")
                    self.model = tf.keras.models.load_model(path)
                    
                    # 加载备用标准化参数
                    if os.path.exists('scaler_mean.npy') and os.path.exists('scaler_scale.npy'):
                        self.feature_mean = np.load('scaler_mean.npy')
                        self.feature_scale = np.load('scaler_scale.npy')
                        self.loaded = True
                        print(f"备用模型加载成功: {path}")
                        return True
            
            print("无法加载任何备用模型")
            return False
            
        except Exception as e:
            print(f"加载备用模型失败: {e}")
            return False
    
    def preprocess_landmarks(self, landmarks):
        """预处理手部关键点"""
        try:
            # 提取增强特征
            features = extract_realtime_features(landmarks)
            
            # 标准化特征
            # 确保特征和标准化参数维度匹配
            if len(features) <= len(self.feature_mean):
                normalized_features = (features - self.feature_mean[:len(features)]) / self.feature_scale[:len(features)]
                return normalized_features.reshape(1, -1)
            else:
                # 如果维度不匹配，截断特征
                print("特征维度不匹配，进行截断")
                truncated_features = features[:len(self.feature_mean)]
                normalized_features = (truncated_features - self.feature_mean) / self.feature_scale
                return normalized_features.reshape(1, -1)
                
        except Exception as e:
            print(f"特征预处理失败: {e}")
            # 应急处理 - 返回零向量
            # 获取模型的输入维度
            input_shape = self.model.layers[0].input_shape[1]
            return np.zeros((1, input_shape))
    
    def predict_gesture(self, preprocessed_data):
        """预测手势"""
        if not self.loaded:
            return None, 0, []
            
        try:
            # 进行预测
            prediction = self.model.predict(preprocessed_data, verbose=0)
            gesture_id = np.argmax(prediction[0])
            confidence = prediction[0][gesture_id]
            
            return gesture_id, confidence, prediction[0]
        
        except Exception as e:
            print(f"预测失败: {e}")
            return None, 0, []
    
    def run(self):
        """运行手势识别"""
        # 检查模型是否加载成功
        if not self.loaded:
            print("错误: 模型未成功加载，无法执行识别")
            return
        
        # 初始化摄像头
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("错误: 无法打开摄像头")
            return
        
        # 设置摄像头分辨率
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # FPS计算
        fps_counter = 0
        fps_start_time = time.time()
        fps = 0
        
        print("开始手势识别. 按ESC键退出.")
        
        while cap.isOpened():
            # 读取帧
            success, image = cap.read()
            if not success:
                print("无法读取摄像头画面")
                break
                
            # 计算FPS
            fps_counter += 1
            current_time = time.time()
            if (current_time - fps_start_time) > 1:
                fps = fps_counter / (current_time - fps_start_time)
                fps_counter = 0
                fps_start_time = current_time
            
            # 为了提高性能，可选择先缩小图像
            image = cv2.resize(image, (640, 480))
            
            # 翻转图像以更好地反映真实手势
            image = cv2.flip(image, 1)
            
            # 转换为RGB进行MediaPipe处理
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.hands.process(image_rgb)
            
            # 显示FPS
            cv2.putText(image, f"FPS: {int(fps)}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 显示指导信息
            guide_text = "请将手掌放在摄像头视野范围内"
            image = put_chinese_text(image, guide_text, (10, 440), font_size=20, color=(255, 255, 255))
            
            # 处理检测结果
            if results.multi_hand_landmarks:
                # 获取第一个手的关键点
                hand_landmarks = results.multi_hand_landmarks[0]
                
                # 绘制手部关键点
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                
                try:
                    # 预处理关键点
                    preprocessed_data = self.preprocess_landmarks(hand_landmarks.landmark)
                    
                    # 预测手势
                    gesture_id, confidence, all_probabilities = self.predict_gesture(preprocessed_data)
                    
                    if gesture_id is not None:
                        # 添加到历史记录
                        self.gesture_history.append(gesture_id)
                        self.confidence_history.append(confidence)
                        
                        # 计算稳定的预测
                        if self.gesture_history:
                            # 最频繁出现的手势
                            from collections import Counter
                            counter = Counter(self.gesture_history)
                            most_common = counter.most_common(1)[0][0]
                            
                            # 平均置信度
                            avg_confidence = np.mean(self.confidence_history)
                            
                            # 判断是否有足够的置信度
                            if avg_confidence > self.confidence_threshold:
                                # 获取手势名称
                                gesture_name = GESTURE_NAMES.get(most_common, "未知")
                                
                                # 显示预测结果
                                result_text = f"手势: {gesture_name} ({avg_confidence:.2f})"
                                image = put_chinese_text(image, result_text, (50, 60))
                                
                                # 显示各个手势的概率
                                prob_y_start = 90
                                for i in range(len(all_probabilities)):
                                    # 使用不同颜色显示当前识别的手势
                                    color = (0, 0, 255) if i == most_common else (255, 0, 0)
                                    prob_text = f"手势 {GESTURE_NAMES.get(i, i)}: {all_probabilities[i]:.3f}"
                                    image = put_chinese_text(image, prob_text, 
                                                           (50, prob_y_start + i*30), 
                                                           font_size=22, color=color)
                            else:
                                # 置信度不足
                                text = f"置信度不足 ({avg_confidence:.2f}), 请调整手势"
                                image = put_chinese_text(image, text, (50, 60), color=(0, 128, 255))
                
                except Exception as e:
                    # 显示错误消息
                    error_text = f"识别出错: {str(e)}"
                    image = put_chinese_text(image, error_text, (50, 60), color=(0, 0, 255))
                    print(f"处理过程出错: {e}")
            else:
                # 未检测到手
                self.gesture_history.clear()  # 清空历史
                self.confidence_history.clear()
                
                text = "未检测到手部"
                image = put_chinese_text(image, text, (50, 60), color=(0, 0, 255))
            
            # 显示图像
            cv2.imshow('手势识别', image)
            
            # 按ESC键退出
            key = cv2.waitKey(5)
            if key == 27:  # ESC
                break
        
        # 释放资源
        self.hands.close()
        cap.release()
        cv2.destroyAllWindows()

# 主函数
if __name__ == "__main__":
    recognizer = GestureRecognizer()
    recognizer.run() 