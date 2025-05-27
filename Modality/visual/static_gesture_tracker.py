import cv2
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import time
import logging
import os
import mediapipe as mp
from collections import deque
import tensorflow as tf

# 配置日志
logging.basicConfig(
    level=logging.DEBUG if os.environ.get('MODALITY_DEBUG', '0') == '1' else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='static_gesture_tracker.log',
    filemode='w'
)
logger = logging.getLogger('GestureTracker')

DEBUG = os.environ.get('MODALITY_DEBUG', '0') == '1'

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

from .base_visual import BaseVisualModality, VisualState
from ..core.error_codes import (
    SUCCESS, MEDIAPIPE_INITIALIZATION_FAILED, RUNTIME_ERROR, 
    MODEL_LOADING_FAILED, MODEL_NOT_FOUND
)

# 手势名称
GESTURE_NAMES = {
    0: "0", 1: "1", 2: "2", 3: "3", 4: "4",
    5: "5", 6: "6", 7: "7", 8: "8", 9: "9",
    10: "ignore"
}

class GestureState(VisualState):
    """手势状态类"""
    
    def __init__(self, frame=None, timestamp=None):
        super().__init__(frame, timestamp)
        self.detections = {
            "gesture": {
                "id": 10,
                "name": "ignore",
                "confidence": 0.0,
                "detected": False,
                "landmarks": [],
                "all_probabilities": [],
                "stability": 0.0
            }
        }
    
    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        return result

def normalize_landmarks(landmarks):
    """将关键点标准化到单位框内，以提高稳定性"""
    # 提取坐标
    x_coords = np.array([lm[0] for lm in landmarks])
    y_coords = np.array([lm[1] for lm in landmarks])
    z_coords = np.array([lm[2] for lm in landmarks])
    
    # 确定边界框
    min_x, max_x = np.min(x_coords), np.max(x_coords)
    min_y, max_y = np.min(y_coords), np.max(y_coords)
    min_z, max_z = np.min(z_coords), np.max(z_coords)
    
    # 计算宽度和高度，添加小值防止除以零
    width = max(max_x - min_x, 1e-5)
    height = max(max_y - min_y, 1e-5)
    depth = max(max_z - min_z, 1e-5)
    
    # 归一化到[0,1]范围
    x_normalized = (x_coords - min_x) / width
    y_normalized = (y_coords - min_y) / height
    z_normalized = (z_coords - min_z) / depth
    
    # 合并回landmarks格式
    normalized_landmarks = []
    for i in range(len(landmarks)):
        normalized_landmarks.append([x_normalized[i], y_normalized[i], z_normalized[i]])
    
    return normalized_landmarks

def calculate_angles(landmarks):
    """计算手指间的关键角度，作为特征"""
    angles = []
    
    # 手指基部关节 (指根)
    finger_bases = [1, 5, 9, 13, 17]
    # 手指关节 (中间关节)
    finger_joints = [2, 6, 10, 14, 18]
    # 手指顶部 (指尖)
    finger_tips = [4, 8, 12, 16, 20]
    
    # 手腕位置
    wrist = np.array(landmarks[0])
    
    # 计算手指弯曲角度
    for base, joint, tip in zip(finger_bases, finger_joints, finger_tips):
        # 向量1: 从基部到关节
        v1 = np.array(landmarks[joint]) - np.array(landmarks[base])
        # 向量2: 从关节到指尖
        v2 = np.array(landmarks[tip]) - np.array(landmarks[joint])
        
        # 计算角度 (使用点积)
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # 避免数值误差
        angle = np.arccos(cos_angle)
        angles.append(angle)
    
    # 计算指尖之间的相对位置
    for i in range(5):
        for j in range(i+1, 5):
            tip_i = np.array(landmarks[finger_tips[i]])
            tip_j = np.array(landmarks[finger_tips[j]])
            
            # 计算距离
            distance = np.linalg.norm(tip_i - tip_j)
            angles.append(distance)
    
    # 计算拇指与其它指尖的夹角
    thumb_tip = np.array(landmarks[4])
    for i in range(1, 5):
        tip = np.array(landmarks[finger_tips[i]])
        v1 = thumb_tip - wrist
        v2 = tip - wrist
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        angles.append(angle)
    
    return angles

def extract_features(landmarks):
    """从手部关键点提取强健的特征"""
    # 归一化关键点
    normalized_landmarks = normalize_landmarks(landmarks)
    
    # 提取手指角度特征
    angle_features = calculate_angles(normalized_landmarks)
    
    # 展平关键点坐标
    coordinate_features = []
    for lm in normalized_landmarks:
        coordinate_features.extend(lm)
    
    # 合并所有特征
    features = np.concatenate([coordinate_features, angle_features])
    
    return features

class GestureTracker(BaseVisualModality):
    """
    手势跟踪器，识别用户的手部姿势和手势
    """
    
    def __init__(self, name: str = "static_gesture_tracker", source: int = 0, 
                 width: int = 640, height: int = 480,
                 min_detection_confidence: float = 0.7, 
                 min_tracking_confidence: float = 0.7,
                 model_path: str = 'Modality/models/static_gesture_recognition/model_output/gesture_model.h5',
                 feature_mean_path: str = 'Modality/models/static_gesture_recognition/model_data/feature_mean.npy',
                 feature_scale_path: str = 'Modality/models/static_gesture_recognition/model_data/feature_scale.npy',
                 confidence_threshold: float = 0.75,
                 stability_threshold: float = 0.7,
                 min_history_size: int = 5,
                 debug: bool = DEBUG):
        """
        初始化手势跟踪器
        
        Args:
            name: 模态名称
            source: 视频源，可以是摄像头ID或视频文件路径
            width: 图像宽度
            height: 图像高度
            min_detection_confidence: 检测置信度阈值
            min_tracking_confidence: 跟踪置信度阈值
            model_path: 手势识别模型路径
            feature_mean_path: 特征均值文件路径
            feature_scale_path: 特征缩放文件路径
            confidence_threshold: 手势识别置信度阈值
            stability_threshold: 手势稳定性阈值
            min_history_size: 最小历史记录大小
            debug: 是否启用调试模式
        """
        super().__init__(name, source, width, height)
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.model_path = model_path
        self.feature_mean_path = feature_mean_path
        self.feature_scale_path = feature_scale_path
        self.confidence_threshold = confidence_threshold
        self.stability_threshold = stability_threshold
        self.min_history_size = min_history_size
        self.debug = debug
        
        # 初始化状态
        self.model = None
        self.feature_mean = None
        self.feature_scale = None
        self.loaded = False
        self.hands = None
        
        # 手势历史记录
        self.gesture_history = deque(maxlen=10)
        self.confidence_history = deque(maxlen=10)
        
        logger.info(f"手势跟踪器初始化完成，调试模式：{self.debug}")
    
    def initialize(self) -> int:
        """
        初始化摄像头和MediaPipe手部跟踪
        
        Returns:
            int: 错误码，0表示成功，其他值表示失败
        """
        result = super().initialize()
        if result != SUCCESS:
            logger.error("基础视觉模态初始化失败")
            return result

        try:
            # 设置MediaPipe Hands
            self.hands = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence
            )
            logger.info("MediaPipe手部检测初始化成功")
        except Exception as e:
            logger.error(f"MediaPipe初始化失败: {str(e)}")
            return MEDIAPIPE_INITIALIZATION_FAILED
        
        # 加载模型和参数
        try:
            # 检查模型文件
            if not os.path.exists(self.model_path):
                logger.error(f"错误: 模型文件不存在 {self.model_path}")
                return MODEL_NOT_FOUND
                
            # 加载模型
            self.model = tf.keras.models.load_model(self.model_path)
            logger.info("模型加载成功!")
            
            # 加载标准化参数
            self.feature_mean = np.load(self.feature_mean_path)
            self.feature_scale = np.load(self.feature_scale_path)
            logger.info("标准化参数加载成功!")
            
            self.loaded = True
        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}")
            return MODEL_LOADING_FAILED
            
        return SUCCESS
    
    def _process_frame(self, frame: np.ndarray) -> GestureState:
        """
        处理图像帧，识别手势
        
        Args:
            frame: 输入图像帧
            
        Returns:
            GestureState: 手势状态
        """
        state = GestureState(frame=frame)
        
        if not self.loaded:
            return state
        
        try:
            # 调整图像大小，提高性能
            frame = cv2.resize(frame, (self.width, self.height))
            
            # 翻转图像以更好地反映真实手势
            frame = cv2.flip(frame, 1)
            
            # 转换为RGB进行MediaPipe处理
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            results = self.hands.process(rgb_frame)
            rgb_frame.flags.writeable = True
            
            # 处理检测结果
            if results.multi_hand_landmarks:
                # 获取第一个手的关键点
                hand_landmarks = results.multi_hand_landmarks[0]
                
                # 绘制手部关键点
                if self.debug:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
                
                try:
                    # 检查手部质量
                    quality_ok = self._check_hand_quality(hand_landmarks.landmark)
                    if not quality_ok:
                        # 手部质量不佳，视为"ignore"
                        self.gesture_history.append(10)  # 添加"ignore"到历史
                        self.confidence_history.append(0.3)  # 低置信度
                        
                        state.detections["gesture"]["id"] = 10
                        state.detections["gesture"]["name"] = "ignore"
                        state.detections["gesture"]["confidence"] = 0.3
                        state.detections["gesture"]["detected"] = True
                        state.detections["gesture"]["stability"] = 0.0
                        
                        if self.debug:
                            logger.debug("手部姿势不清晰")
                    else:
                        # 预处理关键点
                        preprocessed_data = self._preprocess_landmarks(hand_landmarks.landmark)
                        
                        # 预测手势
                        gesture_id, confidence, all_probabilities = self._predict_gesture(preprocessed_data)
                        
                        if gesture_id is not None:
                            # 添加到历史记录
                            self.gesture_history.append(gesture_id)
                            self.confidence_history.append(confidence)
                    
                    # 计算稳定的预测
                    if len(self.gesture_history) >= self.min_history_size:
                        # 计算手势稳定性
                        from collections import Counter
                        counter = Counter(self.gesture_history)
                        most_common, most_common_count = counter.most_common(1)[0]
                        stability = most_common_count / len(self.gesture_history)
                        
                        # 平均置信度
                        avg_confidence = np.mean(self.confidence_history)
                        
                        # 只有当手势足够稳定且置信度高时才显示确定的手势
                        if stability >= self.stability_threshold and avg_confidence >= self.confidence_threshold and most_common != 10:
                            # 获取手势名称
                            gesture_name = GESTURE_NAMES.get(most_common, "unknown")
                            
                            # 更新状态
                            state.detections["gesture"]["id"] = most_common
                            state.detections["gesture"]["name"] = gesture_name
                            state.detections["gesture"]["confidence"] = float(avg_confidence)
                            state.detections["gesture"]["detected"] = True
                            state.detections["gesture"]["all_probabilities"] = all_probabilities.tolist()
                            state.detections["gesture"]["stability"] = float(stability)
                            
                            if self.debug:
                                logger.debug(f"检测到手势: {gesture_name}, 置信度: {avg_confidence:.2f}, 稳定性: {stability:.2f}")
                        else:
                            # 不够稳定或置信度低，显示为ignore
                            state.detections["gesture"]["id"] = 10
                            state.detections["gesture"]["name"] = "ignore"
                            state.detections["gesture"]["confidence"] = float(avg_confidence)
                            state.detections["gesture"]["detected"] = True
                            if "all_probabilities" in locals():
                                state.detections["gesture"]["all_probabilities"] = all_probabilities.tolist()
                            state.detections["gesture"]["stability"] = float(stability)
                            
                            if self.debug:
                                logger.debug(f"手势不稳定或置信度低: 稳定性 {stability:.2f}, 置信度 {avg_confidence:.2f}")
                    
                    # 存储手部关键点
                    landmarks_list = []
                    for lm in hand_landmarks.landmark:
                        landmarks_list.append([lm.x, lm.y, lm.z])
                    state.detections["gesture"]["landmarks"] = landmarks_list
                
                except Exception as e:
                    logger.error(f"处理手部关键点时出错: {str(e)}")
            else:
                # 未检测到手
                self.gesture_history.clear()  # 清空历史
                self.confidence_history.clear()
                if self.debug:
                    logger.debug("未检测到手部")
        
        except Exception as e:
            logger.error(f"处理帧时出错: {str(e)}")
        
        return state
    
    def _check_hand_quality(self, landmarks):
        """检查手部姿势的质量"""
        try:
            # 1. 检查手是否完全在画面中
            for lm in landmarks:
                if not (0.05 < lm.x < 0.95 and 0.05 < lm.y < 0.95):
                    if self.debug:
                        logger.debug("手部不完全在画面内")
                    return False
            
            # 2. 计算手指伸直程度 - 检查关键的关节角度
            # 获取手掌关键点
            wrist = np.array([landmarks[0].x, landmarks[0].y, landmarks[0].z])
            thumb_tip = np.array([landmarks[4].x, landmarks[4].y, landmarks[4].z])
            index_tip = np.array([landmarks[8].x, landmarks[8].y, landmarks[8].z])
            middle_tip = np.array([landmarks[12].x, landmarks[12].y, landmarks[12].z])
            ring_tip = np.array([landmarks[16].x, landmarks[16].y, landmarks[16].z])
            pinky_tip = np.array([landmarks[20].x, landmarks[20].y, landmarks[20].z])
            
            # 计算手部大小 - 手掌宽度
            index_mcp = np.array([landmarks[5].x, landmarks[5].y, landmarks[5].z])
            pinky_mcp = np.array([landmarks[17].x, landmarks[17].y, landmarks[17].z])
            palm_width = np.linalg.norm(index_mcp - pinky_mcp)
            
            # 如果手部太小，可能是远处的手，不够清晰
            if palm_width < 0.08:
                if self.debug:
                    logger.debug(f"手部尺寸太小: {palm_width:.3f}")
                return False
            
            # 3. 检查手的稳定性 - 通过连续帧之间的位置变化
            # 这部分在外部通过历史记录实现
                
            return True
            
        except Exception as e:
            logger.error(f"手部质量检查失败: {str(e)}")
            return False
    
    def _preprocess_landmarks(self, landmarks):
        """预处理手部关键点"""
        try:
            # 转换MediaPipe格式到我们的格式
            landmark_list = []
            for lm in landmarks:
                landmark_list.append([lm.x, lm.y, lm.z])
            
            # 提取特征
            features = extract_features(landmark_list)
            
            # 标准化特征
            # 确保特征和标准化参数维度匹配
            if len(features) <= len(self.feature_mean):
                normalized_features = (features - self.feature_mean[:len(features)]) / self.feature_scale[:len(features)]
                return normalized_features.reshape(1, -1)
            else:
                # 如果维度不匹配，截断特征
                if self.debug:
                    logger.debug("特征维度不匹配，进行截断")
                truncated_features = features[:len(self.feature_mean)]
                normalized_features = (truncated_features - self.feature_mean) / self.feature_scale
                return normalized_features.reshape(1, -1)
                
        except Exception as e:
            logger.error(f"特征预处理失败: {str(e)}")
            # 应急处理 - 返回零向量
            # 获取模型的输入维度
            input_shape = self.model.layers[0].input_shape[1]
            return np.zeros((1, input_shape))
    
    def _predict_gesture(self, preprocessed_data):
        """预测手势"""
        if not self.loaded:
            return None, 0, []
            
        try:
            # 进行预测
            prediction = self.model.predict(preprocessed_data, verbose=0)
            gesture_id = np.argmax(prediction[0])
            confidence = prediction[0][gesture_id]
            
            # 如果最高置信度低于阈值，则识别为"ignore"
            if confidence < self.confidence_threshold:
                if self.debug:
                    logger.debug(f"置信度低于阈值: {confidence:.3f} < {self.confidence_threshold}")
                return 10, confidence, prediction[0]  # 返回"ignore"类别
            
            return gesture_id, confidence, prediction[0]
        
        except Exception as e:
            logger.error(f"预测失败: {str(e)}")
            return None, 0, []
    
    def shutdown(self) -> int:
        """
        关闭手势跟踪
        
        Returns:
            int: 错误码，0表示成功，其他值表示失败
        """
        try:
            if self.hands is not None:
                self.hands.close()
            
            # 清空历史记录
            self.gesture_history.clear()
            self.confidence_history.clear()
            
            # 调用父类方法关闭视频源
            result = super().shutdown()
            
            logger.info("手势跟踪器已关闭")
            return result
        except Exception as e:
            logger.error(f"关闭手势跟踪器时出错: {str(e)}")
            return RUNTIME_ERROR 
    
    def get_key_info(self) -> str:
        """
        获取模态的关键信息

        Returns:
            str: 模态的关键信息
        """
        gesture_dir = {"0": "握拳", "5": "摇手", "6": "竖起大拇指"}
        key_info = None
        state = self.update()
        #print(f"是否识别到手: {state.detections["gesture"]["detected"]}")
        if state and state.detections["gesture"]["detected"]:
            key_info = state.detections["gesture"]["name"]
            #print(f"手部初始识别结果: {key_info}")
            if key_info in gesture_dir:
                key_info = gesture_dir[key_info]
        #print(f"手部识别结果: {key_info}")
        return key_info