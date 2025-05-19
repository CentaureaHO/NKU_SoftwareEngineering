import cv2
import numpy as np
from typing import Dict, Any, List, Tuple
import math
import logging
import os
import time
import threading
from collections import deque
import tensorflow as tf
import pickle
import json
import sys

logging.basicConfig(
    level=logging.DEBUG if os.environ.get('MODALITY_DEBUG', '0') == '1' else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='head_tracker_gru.log',
    filemode='w'
)
logger = logging.getLogger('HeadTrackerGRU')

DEBUG = os.environ.get('MODALITY_DEBUG', '0') == '1'

import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Modality.visual.base_visual import BaseVisualModality
from Modality.visual.head_pose_common import HeadPoseParams, HeadPoseState, euclidean_dist, rotation_matrix_to_angles
from Modality.core.error_codes import (
    SUCCESS, MEDIAPIPE_INITIALIZATION_FAILED, RUNTIME_ERROR,
    MODEL_LOADING_FAILED
)
from Modality.visual.base_visual import BaseVisualModality, VisualState

# 导入视线方向常量
GAZE_DIRECTION_CENTER = "center"
GAZE_DIRECTION_LEFT = "left"
GAZE_DIRECTION_RIGHT = "right"
GAZE_DIRECTION_UP = "up"
GAZE_DIRECTION_DOWN = "down"
GAZE_DIRECTION_UP_LEFT = "up_left"
GAZE_DIRECTION_UP_RIGHT = "up_right"
GAZE_DIRECTION_DOWN_LEFT = "down_left"
GAZE_DIRECTION_DOWN_RIGHT = "down_right"

class GazeParams:
    """视线方向检测的参数常量"""
    # 眼睛关键点索引
    LEFT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    RIGHT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    LEFT_IRIS_INDICES = [468, 469, 470, 471, 472]
    RIGHT_IRIS_INDICES = [473, 474, 475, 476, 477]
    
    # 视线方向检测参数
    HORIZONTAL_RATIO_THRESHOLD = 0.35  # 减小以增强水平敏感度
    VERTICAL_RATIO_THRESHOLD = 0.38    
    CENTER_THRESHOLD = 0.40            # 调整中心区域以平衡灵敏度
    UP_THRESHOLD_STRICTER = -0.55      # 增大以降低向上看的敏感度
    DOWN_THRESHOLD_STRICTER = 0.55     # 减小以降低向下看的敏感度
    VERTICAL_OFFSET = 0.03             # 垂直偏移
    
    # 瞳孔检测参数
    PUPIL_DETECTION_THRESHOLD = 0.01   # 瞳孔检测面积比例阈值
    EYE_SPHERE_RADIUS = 15.0           # 眼球球面模型半径
    DEPTH_SCALE_FACTOR = 0.003         # 深度缩放因子
    
    # 平滑参数
    SMOOTHING_WINDOW_SIZE = 10         # 平滑窗口大小
    POSITION_ALPHA = 0.25              # 位置平滑系数
    DIRECTION_ALPHA = 0.3              # 方向平滑系数
    
    # 视线一致性检测阈值
    EYE_CONSISTENCY_THRESHOLD = 0.25   # 左右眼一致性检查阈值
    VERTICAL_CONSISTENCY_WEIGHT = 0.7  # 垂直方向一致性权重
    HORIZONTAL_CONSISTENCY_WEIGHT = 0.3 # 水平方向一致性权重
    
    # 组合视线计算权重
    LEFT_EYE_WEIGHT = 0.5              # 左眼权重
    RIGHT_EYE_WEIGHT = 0.5             # 右眼权重

    PITCH_COMPENSATION_FACTOR = 0.3    # 减小以降低垂直敏感度


class HeadPoseState(VisualState):
    """头部姿态状态类，扩展自VisualState，新增视线方向检测"""
    
    def __init__(self, frame=None, timestamp=None):
        super().__init__(frame, timestamp)
        self.detections = {
            "head_pose": {
                "pitch": 0.0,       # 俯仰角（点头）
                "yaw": 0.0,         # 偏航角（左右转头）
                "roll": 0.0,        # 翻滚角（头部倾斜）
                "detected": False,  # 是否检测到人脸
                "landmarks": []     # 关键点列表
            },
            "head_movement": {
                "is_nodding": False,        # 是否正在点头
                "is_shaking": False,        # 是否正在摇头
                "nod_confidence": 0.0,      # 点头动作的置信度
                "shake_confidence": 0.0,    # 摇头动作的置信度
                "status": HeadPoseParams.STATUS_STATIONARY,  # 当前状态
                "status_confidence": 0.0    # 当前状态的置信度
            },
            "gaze_direction": {
                "direction": GAZE_DIRECTION_CENTER,  # 视线方向
                "confidence": 0.0,                   # 置信度
                "horizontal_ratio": 0.0,             # 水平比例(-1到1)
                "vertical_ratio": 0.0,               # 垂直比例(-1到1)
                "left_eye": {
                    "iris_position": (0.0, 0.0),     # 左眼虹膜位置比例
                    "eye_landmarks": []              # 左眼关键点
                },
                "right_eye": {
                    "iris_position": (0.0, 0.0),     # 右眼虹膜位置比例
                    "eye_landmarks": []              # 右眼关键点
                }
            }
        }
    
    def __repr__(self):
        return f"<HeadPoseState(pitch={self.detections['head_pose']['pitch']}, yaw={self.detections['head_pose']['yaw']}, roll={self.detections['head_pose']['roll']}, is_nodding={self.detections['head_movement']['is_nodding']}, is_shaking={self.detections['head_movement']['is_shaking']}, gaze_direction={self.detections['gaze_direction']['direction']})>"
    
class HeadPoseTrackerGRU(BaseVisualModality):
    """
    基于GRU深度学习模型的头部姿态跟踪器，集成了视线方向检测功能
    
    使用训练好的GRU模型检测头部姿势和动作，包括静止、点头、摇头等，
    同时追踪视线方向
    """
    
    def __init__(self, name: str = "head_pose_tracker_gru", source: int = 0, 
                 width: int = 640, height: int = 480,
                 min_detection_confidence: float = 0.5, 
                 min_tracking_confidence: float = 0.5,
                 model_dir: str = None,
                 debug: bool = DEBUG):
        """
        初始化基于GRU模型的头部姿态跟踪器
        
        Args:
            name: 模态名称
            source: 视频源，可以是摄像头ID或视频文件路径
            width: 图像宽度
            height: 图像高度
            min_detection_confidence: 检测置信度阈值
            min_tracking_confidence: 跟踪置信度阈值
            model_dir: 模型目录路径
            debug: 是否启用调试模式
        """
        super().__init__(name, source, width, height)
        
        if model_dir is None:
            self.model_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                "models", "headpose"
            )
        else:
            self.model_dir = model_dir
            
        self.mp_face_options = {
            'max_num_faces': 1,
            'refine_landmarks': True,
            'min_detection_confidence': min_detection_confidence,
            'min_tracking_confidence': min_tracking_confidence,
        }
        
        self.face_3d_coords = np.array([
            [285, 528, 200],  # 鼻子
            [285, 371, 152],  # 鼻子下方
            [197, 574, 128],  # 左脸边缘
            [173, 425, 108],  # 左眼附近
            [360, 574, 128],  # 右脸边缘
            [391, 425, 108]   # 右眼附近
        ], dtype=np.float64)

        self.model = None
        self.scaler = None
        self.config = None
        self.window_size = None
        self.stride = None
        self.feature_dim = None
        self.gesture_mapping = None
        self.inverse_gesture_mapping = None
        
        self.features_queue = deque(maxlen=HeadPoseParams.HISTORY_LEN)
        
        self.left_h_ratio_history = deque(maxlen=GazeParams.SMOOTHING_WINDOW_SIZE)
        self.left_v_ratio_history = deque(maxlen=GazeParams.SMOOTHING_WINDOW_SIZE)
        self.right_h_ratio_history = deque(maxlen=GazeParams.SMOOTHING_WINDOW_SIZE)
        self.right_v_ratio_history = deque(maxlen=GazeParams.SMOOTHING_WINDOW_SIZE)
        self.left_pupil_center_history = deque(maxlen=GazeParams.SMOOTHING_WINDOW_SIZE)
        self.right_pupil_center_history = deque(maxlen=GazeParams.SMOOTHING_WINDOW_SIZE)
        self.left_gaze_direction_history = deque(maxlen=GazeParams.SMOOTHING_WINDOW_SIZE)
        self.right_gaze_direction_history = deque(maxlen=GazeParams.SMOOTHING_WINDOW_SIZE)
        
        self.h_ratio_variance = 0.1
        self.v_ratio_variance = 0.1
        
        self._processing_thread = None
        self._stop_event = threading.Event()
        
        self._latest_state = HeadPoseState()
        self._state_lock = threading.Lock()
        
        self._last_status_update = time.time()
        
        self._current_is_nodding = False
        self._current_is_shaking = False
        self._current_status = HeadPoseParams.STATUS_STATIONARY
        self._current_status_confidence = 0.0
        
        self.face_mesh = None
        self.debug = debug
    
    def initialize(self) -> int:
        """
        初始化摄像头、MediaPipe人脸网格和GRU模型
        
        Returns:
            int: 错误码，0表示成功，其他值表示失败
        """
        result = super().initialize()
        if result != SUCCESS:
            logger.error("基础视觉模态初始化失败")
            return result

        try:
            # 设置MediaPipe
            self.face_mesh = mp_face_mesh.FaceMesh(
                max_num_faces=self.mp_face_options['max_num_faces'],
                refine_landmarks=self.mp_face_options['refine_landmarks'],
                min_detection_confidence=self.mp_face_options['min_detection_confidence'],
                min_tracking_confidence=self.mp_face_options['min_tracking_confidence'],
            )
            logger.info("MediaPipe人脸网格初始化成功")
            
            # 加载模型和配置
            result = self._load_model()
            if result != SUCCESS:
                logger.error("模型加载失败")
                return result
            
            return SUCCESS
        except Exception as e:
            logger.error(f"初始化失败: {str(e)}")
            return MEDIAPIPE_INITIALIZATION_FAILED
    
    def _load_model(self) -> int:
        """
        加载GRU模型、缩放器和配置
        
        Returns:
            int: 错误码，0表示成功，其他值表示失败
        """
        try:
            # 找到模型文件
            model_path = os.path.join(self.model_dir, "model.h5")
            scaler_path = os.path.join(self.model_dir, "scaler.pkl")
            config_path = os.path.join(self.model_dir, "config.json")
            
            if not os.path.exists(model_path):
                logger.error(f"模型文件不存在: {model_path}")
                return MODEL_LOADING_FAILED
                
            if not os.path.exists(scaler_path):
                logger.error(f"缩放器文件不存在: {scaler_path}")
                return MODEL_LOADING_FAILED
                
            if not os.path.exists(config_path):
                logger.error(f"配置文件不存在: {config_path}")
                return MODEL_LOADING_FAILED
            
            # 加载模型
            logger.info(f"正在加载模型: {model_path}")
            self.model = tf.keras.models.load_model(model_path)
            
            # 加载缩放器
            logger.info(f"正在加载缩放器: {scaler_path}")
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            
            # 加载配置
            logger.info(f"正在加载配置: {config_path}")
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
            
            # 设置模型参数
            self.window_size = self.config.get("window_size", HeadPoseParams.HISTORY_LEN)
            self.stride = self.config.get("stride", 5)
            self.feature_dim = self.config.get("feature_dim", 30)
            self.gesture_mapping = self.config.get("gesture_mapping", {
                "stationary": 0, "nodding": 1, "shaking": 2, "other": 3
            })
            self.inverse_gesture_mapping = {v: k for k, v in self.gesture_mapping.items()}
            
            # 初始化特征队列
            self.features_queue = deque(maxlen=self.window_size)
            
            logger.info(f"模型加载成功，窗口大小: {self.window_size}, 特征维度: {self.feature_dim}")
            logger.info(f"姿势映射: {self.gesture_mapping}")
            
            return SUCCESS
        except Exception as e:
            logger.error(f"加载模型时出错: {str(e)}")
            return MODEL_LOADING_FAILED
    
    def _processing_loop(self):
        """处理线程的主循环，负责连续处理视频帧"""
        logger.info("处理线程已启动")
        
        if self.capture is None:
            logger.error("视频源未初始化")
            return
        
        frame_interval = 1.0 / HeadPoseParams.VIDEO_FPS
        
        while not self._stop_event.is_set():
            start_time = time.time()
            
            ret, frame = self.capture.read()
            if not ret:
                if self.is_file_source and self.loop_video:
                    self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    logger.error("无法获取视频帧")
                    break
            
            state = self._process_frame(frame)
            with self._state_lock:
                self._latest_state = state
            
            elapsed_time = time.time() - start_time
            sleep_time = max(0, frame_interval - elapsed_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def update(self) -> HeadPoseState:
        """
        获取最新的头部姿态状态
        
        Returns:
            HeadPoseState: 当前头部状态
        """
        if not self._is_running or not self._processing_thread or not self._processing_thread.is_alive():
            logger.warning("处理线程未运行")
            return HeadPoseState()
        
        with self._state_lock:
            return self._latest_state
    
    def _extract_features(self, face_landmarks, frame_shape) -> np.ndarray:
        """
        从人脸关键点提取特征
        
        Args:
            face_landmarks: MediaPipe人脸网格关键点
            frame_shape: 图像帧形状
            
        Returns:
            np.ndarray: 提取的特征向量或None
        """
        h, w = frame_shape[:2]
        
        # 收集关键点
        face_coordination_in_image = []
        nose_point = None
        chin_point = None
        left_ear_point = None
        left_face_point = None
        right_ear_point = None
        right_face_point = None
        all_face_coords = []
        
        for idx, lm in enumerate(face_landmarks.landmark):
            x, y = int(lm.x * w), int(lm.y * h)
            all_face_coords.append((x, y))
            
            if idx in [1, 9, 57, 130, 287, 359]:
                face_coordination_in_image.append([x, y])
            
            # 收集用于点头和摇头检测的点
            if idx == HeadPoseParams.LANDMARK_NOSE:  # 鼻尖
                nose_point = (x, y)
            elif idx == HeadPoseParams.LANDMARK_CHIN:  # 下巴
                chin_point = (x, y)
            elif idx == HeadPoseParams.LANDMARK_LEFT_EAR:  # 左耳
                left_ear_point = (x, y)
            elif idx == HeadPoseParams.LANDMARK_LEFT_FACE:  # 左脸中心点
                left_face_point = (x, y)
            elif idx == HeadPoseParams.LANDMARK_RIGHT_EAR:  # 右耳
                right_ear_point = (x, y)
            elif idx == HeadPoseParams.LANDMARK_RIGHT_FACE:  # 右脸中心点
                right_face_point = (x, y)
        
        if None in [nose_point, chin_point, left_ear_point, left_face_point, right_ear_point, right_face_point]:
            return None
            
        if len(face_coordination_in_image) != 6:
            return None
            
        face_coordination_in_image = np.array(face_coordination_in_image, dtype=np.float64)
        
        # 获取人脸框信息
        all_face_coords = np.array(all_face_coords)
        x_min, y_min = np.min(all_face_coords, axis=0)
        x_max, y_max = np.max(all_face_coords, axis=0)
        box_width = max(x_max - x_min, 1)
        box_height = max(y_max - y_min, 1)
        box_diagonal = np.sqrt(box_width**2 + box_height**2)
        aspect_ratio = box_width / box_height
        
        # 计算头部姿态
        focal_length = 1 * w
        cam_matrix = np.array([
            [focal_length, 0, w / 2],
            [0, focal_length, h / 2],
            [0, 0, 1]
        ])
        dist_matrix = np.zeros((4, 1), dtype=np.float64)
        
        try:
            success, rotation_vec, translation_vec = cv2.solvePnP(
                self.face_3d_coords, face_coordination_in_image,
                cam_matrix, dist_matrix
            )
            
            if not success:
                return None
                
            rotation_matrix, _ = cv2.Rodrigues(rotation_vec)
            angles = rotation_matrix_to_angles(rotation_matrix)
            
            pitch = float(angles[0])  # 俯仰角（点头）
            yaw = float(angles[1])    # 偏航角（左右转头）
            roll = float(angles[2])   # 翻滚角（头部倾斜）
            
            # 计算几何特征
            nose_chin_dist = euclidean_dist(nose_point, chin_point)
            left_cheek_width = euclidean_dist(left_ear_point, left_face_point)
            right_cheek_width = euclidean_dist(right_ear_point, right_face_point)
            
            # 基本特征
            basic_features = np.array([
                pitch,
                yaw,
                roll,
                nose_chin_dist,
                left_cheek_width,
                right_cheek_width
            ], dtype=np.float32)
            
            # 归一化特征
            normalized_features = np.array([
                aspect_ratio,
                box_width / box_diagonal,
                box_height / box_diagonal,
                
                pitch / box_diagonal,
                yaw / box_diagonal,
                roll / box_diagonal,
                
                nose_chin_dist / box_height,
                left_cheek_width / box_width,
                right_cheek_width / box_width
            ], dtype=np.float32)
            
            # 组合特征
            combined_features = np.concatenate([basic_features, normalized_features])
            
            return combined_features
            
        except Exception as e:
            logger.error(f"计算头部姿态和特征时出错: {str(e)}")
            return None
    
    def _predict_head_pose(self) -> Tuple[str, float]:
        """
        使用GRU模型预测头部姿势
        
        Returns:
            Tuple[str, float]: 预测的姿势和置信度
        """
        if len(self.features_queue) < self.window_size:
            return None, 0.0
        
        try:
            # 创建差分特征序列
            diff_seq = []
            features_list = list(self.features_queue)
            
            for i in range(1, len(features_list)):
                combined_features = np.concatenate([
                    features_list[i], 
                    features_list[i] - features_list[i-1]
                ])
                diff_seq.append(combined_features)
            
            # 准备模型输入
            sequence = np.array([diff_seq])
            
            # 缩放特征
            shape = sequence.shape
            sequence_flat = sequence.reshape((-1, self.feature_dim))
            sequence_flat = self.scaler.transform(sequence_flat)
            sequence = sequence_flat.reshape(shape)
            
            # 模型预测
            prediction = self.model.predict(sequence, verbose=0)
            gesture_idx = np.argmax(prediction[0])
            confidence = float(prediction[0][gesture_idx])
            
            # 获取预测的姿势名称
            pose_name = self.inverse_gesture_mapping.get(gesture_idx, HeadPoseParams.STATUS_STATIONARY)
            
            return pose_name, confidence
            
        except Exception as e:
            logger.error(f"预测头部姿势时出错: {str(e)}")
            return HeadPoseParams.STATUS_STATIONARY, 0.0
    
    def _smooth_value(self, value_history, new_value, alpha=None):
        """使用指数移动平均平滑数值"""
        if not value_history:
            return new_value
        
        if alpha is None:
            alpha = GazeParams.POSITION_ALPHA
            
        if isinstance(new_value, tuple):
            prev_x, prev_y = value_history[-1]
            curr_x, curr_y = new_value
            return (alpha * curr_x + (1 - alpha) * prev_x,
                   alpha * curr_y + (1 - alpha) * prev_y)
        else:
            return alpha * new_value + (1 - alpha) * value_history[-1]
    
    def _enhanced_iris_detection(self, eye_region, landmarks):
        """改进的虹膜检测方法，结合图像处理技术提高精度"""
        try:
            if eye_region.size == 0:
                return None
                
            gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
            
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            _, thresh1 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            thresh2 = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY_INV, 11, 2)
            
            thresh = cv2.bitwise_and(thresh1, thresh2)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
            
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            min_area = eye_region.shape[0] * eye_region.shape[1] * GazeParams.PUPIL_DETECTION_THRESHOLD
            valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
            
            if valid_contours:
                best_contour = None
                darkest_value = 255
                
                for contour in valid_contours:
                    mask = np.zeros_like(gray)
                    cv2.drawContours(mask, [contour], 0, 255, -1)
                    mean_val = cv2.mean(gray, mask=mask)[0]
                    
                    if mean_val < darkest_value:
                        darkest_value = mean_val
                        best_contour = contour
                
                if best_contour is not None:
                    M = cv2.moments(best_contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        return (cx, cy)
                    
        except Exception as e:
            logger.debug(f"虹膜检测错误: {str(e)}")
        
        return None
    
    def _calculate_eye_center(self, eye_landmarks):
        """计算眼部中心点"""
        x_coords = [p[0] for p in eye_landmarks]
        y_coords = [p[1] for p in eye_landmarks]
        return (sum(x_coords) / len(x_coords), sum(y_coords) / len(y_coords))
    
    def _calculate_3d_gaze_direction(self, pupil_center, eye_center, image_height):
        """计算3D视线方向向量"""
        if not pupil_center or not eye_center:
            return None
            
        ox, oy = eye_center
        px, py = pupil_center
        dx_2d = px - ox
        dy_2d = py - oy
        
        distance_2d = math.sqrt(dx_2d ** 2 + dy_2d ** 2)
        if distance_2d < 1e-6:
            return (0, 0, 1)
            
        depth_scale = image_height * GazeParams.DEPTH_SCALE_FACTOR
        
        r_squared = GazeParams.EYE_SPHERE_RADIUS ** 2
        d_squared = (distance_2d / depth_scale) ** 2
        
        value = r_squared - d_squared
        dz = math.sqrt(max(0.1, value)) if value >= 0 else 0.1
        
        norm = math.sqrt(dx_2d ** 2 + dy_2d ** 2 + (dz * depth_scale) ** 2)
        
        return (dx_2d / norm, dy_2d / norm, (dz * depth_scale) / norm)
    
    def _extract_eye_region(self, eye_landmarks, frame):
        """提取眼部区域图像"""
        eye_x_coords = [p[0] for p in eye_landmarks]
        eye_y_coords = [p[1] for p in eye_landmarks]
        
        x1 = max(0, int(min(eye_x_coords)) - 5)
        y1 = max(0, int(min(eye_y_coords)) - 5)
        x2 = min(frame.shape[1], int(max(eye_x_coords)) + 5)
        y2 = min(frame.shape[0], int(max(eye_y_coords)) + 5)
        
        if x2 <= x1 or y2 <= y1:
            return None, (x1, y1)
            
        eye_region = frame[y1:y2, x1:x2]
        return eye_region, (x1, y1)
    
    def _calculate_iris_position(self, eye_landmarks, iris_landmarks, frame=None):
        """计算虹膜相对于眼睛的位置，增强版"""
        eye_x_coords = [p[0] for p in eye_landmarks]
        eye_y_coords = [p[1] for p in eye_landmarks]
        
        eye_left = min(eye_x_coords)
        eye_right = max(eye_x_coords)
        eye_top = min(eye_y_coords)
        eye_bottom = max(eye_y_coords)
        
        eye_width = max(eye_right - eye_left, 1e-5)
        eye_height = max(eye_bottom - eye_top, 1e-5)
        
        eye_center = self._calculate_eye_center(eye_landmarks)
        
        iris_center_x = sum([p[0] for p in iris_landmarks]) / len(iris_landmarks)
        iris_center_y = sum([p[1] for p in iris_landmarks]) / len(iris_landmarks)
        iris_center = (iris_center_x, iris_center_y)
        
        if frame is not None:
            eye_region, offset = self._extract_eye_region(eye_landmarks, frame)
            if eye_region is not None and eye_region.size > 0:
                enhanced_iris = self._enhanced_iris_detection(eye_region, 
                    [(p[0] - offset[0], p[1] - offset[1]) for p in eye_landmarks])
                
                if enhanced_iris:
                    iris_center = (enhanced_iris[0] + offset[0], enhanced_iris[1] + offset[1])
                    iris_center_x, iris_center_y = iris_center
        
        horizontal_ratio = 2 * (iris_center_x - eye_left) / eye_width - 1
        vertical_ratio = 2 * (iris_center_y - eye_top) / eye_height - 1
        
        vertical_ratio = vertical_ratio + GazeParams.VERTICAL_OFFSET
        vertical_ratio = max(-1.0, min(1.0, vertical_ratio))
        horizontal_ratio = max(-1.0, min(1.0, horizontal_ratio))
        
        gaze_3d = self._calculate_3d_gaze_direction(
            iris_center, eye_center, frame.shape[0] if frame is not None else 480)
            
        return horizontal_ratio, vertical_ratio, gaze_3d, iris_center
    
    def _check_eye_consistency(self, left_h_ratio, left_v_ratio, right_h_ratio, right_v_ratio):
        """检查左右眼视线方向是否一致"""
        h_diff = abs(left_h_ratio - right_h_ratio)
        v_diff = abs(left_v_ratio - right_v_ratio)
        
        h_consistent = h_diff <= GazeParams.EYE_CONSISTENCY_THRESHOLD
        v_consistent = v_diff <= GazeParams.EYE_CONSISTENCY_THRESHOLD
        
        h_weight = GazeParams.HORIZONTAL_CONSISTENCY_WEIGHT
        v_weight = GazeParams.VERTICAL_CONSISTENCY_WEIGHT
        
        consistency_score = (h_weight * (1 - h_diff / max(1, abs(left_h_ratio) + abs(right_h_ratio))) + 
                           v_weight * (1 - v_diff / max(1, abs(left_v_ratio) + abs(right_v_ratio))))
        
        is_consistent = h_consistent and v_consistent
        
        return is_consistent, consistency_score
    
    def _determine_gaze_direction(self, left_h_ratio, left_v_ratio, left_gaze_3d,
                                 right_h_ratio, right_v_ratio, right_gaze_3d,
                                 head_pitch=0.0, head_yaw=0.0):
        """
        根据两只眼睛的虹膜位置确定视线方向，从摄像机视角判断
        
        Args:
            left_h_ratio, left_v_ratio: 左眼水平和垂直比例
            left_gaze_3d: 左眼3D视线向量
            right_h_ratio, right_v_ratio: 右眼水平和垂直比例
            right_gaze_3d: 右眼3D视线向量
            head_pitch: 头部俯仰角(点头角度)，单位为弧度，正值表示低头，负值表示抬头
            head_yaw: 头部偏航角(左右转头角度)，单位为弧度，正值表示向右，负值表示向左
        """
        is_consistent, consistency_score = self._check_eye_consistency(
            left_h_ratio, left_v_ratio, right_h_ratio, right_v_ratio)
        
        smooth_left_h = self._smooth_value(self.left_h_ratio_history, left_h_ratio)
        smooth_left_v = self._smooth_value(self.left_v_ratio_history, left_v_ratio)
        smooth_right_h = self._smooth_value(self.right_h_ratio_history, right_h_ratio)
        smooth_right_v = self._smooth_value(self.right_v_ratio_history, right_v_ratio)
        
        self.left_h_ratio_history.append(smooth_left_h)
        self.left_v_ratio_history.append(smooth_left_v)
        self.right_h_ratio_history.append(smooth_right_h)
        self.right_v_ratio_history.append(smooth_right_v)
        
        h_ratios = list(self.left_h_ratio_history) + list(self.right_h_ratio_history)
        v_ratios = list(self.left_v_ratio_history) + list(self.right_v_ratio_history)
        if len(h_ratios) >= 3:
            self.h_ratio_variance = np.var(h_ratios) + 0.05
            self.v_ratio_variance = np.var(v_ratios) + 0.05
        
        left_weight = GazeParams.LEFT_EYE_WEIGHT
        right_weight = GazeParams.RIGHT_EYE_WEIGHT
        
        if not is_consistent:
            if abs(smooth_left_v) >= abs(smooth_right_v):
                left_weight = 0.7
                right_weight = 0.3
            else:
                left_weight = 0.3
                right_weight = 0.7
        
        avg_h_ratio = left_weight * smooth_left_h + right_weight * smooth_right_h
        avg_v_ratio = left_weight * smooth_left_v + right_weight * smooth_right_v
        
        mirrored_h_ratio = -avg_h_ratio
        
        h_threshold = max(GazeParams.HORIZONTAL_RATIO_THRESHOLD, self.h_ratio_variance * 2)
        # v_threshold = max(GazeParams.VERTICAL_RATIO_THRESHOLD, self.v_ratio_variance * 2) # Not directly used in final decision logic below, but kept for consistency
        center_threshold = GazeParams.CENTER_THRESHOLD # Updated value will be used here
        
        pitch_compensation = head_pitch * GazeParams.PITCH_COMPENSATION_FACTOR
        camera_v_ratio = avg_v_ratio + pitch_compensation
        
        camera_v_ratio = max(-1.0, min(1.0, camera_v_ratio))

        position_confidence = max(abs(mirrored_h_ratio), abs(camera_v_ratio))
        confidence = position_confidence * consistency_score
        
        if abs(mirrored_h_ratio) <= center_threshold and abs(camera_v_ratio) <= center_threshold:
            return GAZE_DIRECTION_CENTER, confidence, mirrored_h_ratio, camera_v_ratio
            
        if camera_v_ratio <= GazeParams.UP_THRESHOLD_STRICTER:
            if mirrored_h_ratio >= h_threshold:
                return GAZE_DIRECTION_UP_RIGHT, confidence, mirrored_h_ratio, camera_v_ratio
            elif mirrored_h_ratio <= -h_threshold:
                return GAZE_DIRECTION_UP_LEFT, confidence, mirrored_h_ratio, camera_v_ratio
            else:
                return GAZE_DIRECTION_UP, confidence, mirrored_h_ratio, camera_v_ratio
        elif camera_v_ratio >= GazeParams.DOWN_THRESHOLD_STRICTER:
            if mirrored_h_ratio >= h_threshold:
                return GAZE_DIRECTION_DOWN_RIGHT, confidence, mirrored_h_ratio, camera_v_ratio
            elif mirrored_h_ratio <= -h_threshold:
                return GAZE_DIRECTION_DOWN_LEFT, confidence, mirrored_h_ratio, camera_v_ratio
            else:
                return GAZE_DIRECTION_DOWN, confidence, mirrored_h_ratio, camera_v_ratio
        else:
            if mirrored_h_ratio >= h_threshold:
                return GAZE_DIRECTION_RIGHT, confidence, mirrored_h_ratio, camera_v_ratio
            elif mirrored_h_ratio <= -h_threshold:
                return GAZE_DIRECTION_LEFT, confidence, mirrored_h_ratio, camera_v_ratio
            else:
                return GAZE_DIRECTION_CENTER, confidence, mirrored_h_ratio, camera_v_ratio

    def _process_gaze_direction(self, frame, face_landmarks, h, w, head_pitch_rad, head_yaw_rad):
        """处理视线方向检测"""
        gaze_result = {
            "direction": GAZE_DIRECTION_CENTER,
            "confidence": 0.0,
            "horizontal_ratio": 0.0,
            "vertical_ratio": 0.0,
            "left_eye": {
                "iris_position": (0.0, 0.0),
                "eye_landmarks": []
            },
            "right_eye": {
                "iris_position": (0.0, 0.0),
                "eye_landmarks": []
            }
        }
        
        left_eye_landmarks = []
        right_eye_landmarks = []
        left_iris_landmarks = []
        right_iris_landmarks = []
        
        for idx, landmark in enumerate(face_landmarks.landmark):
            x, y, z = landmark.x, landmark.y, landmark.z
            pixel_x, pixel_y = int(x * w), int(y * h)

            if idx in GazeParams.LEFT_EYE_INDICES:
                left_eye_landmarks.append((pixel_x, pixel_y))
            if idx in GazeParams.RIGHT_EYE_INDICES:
                right_eye_landmarks.append((pixel_x, pixel_y))
            if idx in GazeParams.LEFT_IRIS_INDICES:
                left_iris_landmarks.append((pixel_x, pixel_y))
            if idx in GazeParams.RIGHT_IRIS_INDICES:
                right_iris_landmarks.append((pixel_x, pixel_y))

        if (len(left_eye_landmarks) == len(GazeParams.LEFT_EYE_INDICES) and
            len(right_eye_landmarks) == len(GazeParams.RIGHT_EYE_INDICES) and
            len(left_iris_landmarks) == len(GazeParams.LEFT_IRIS_INDICES) and
            len(right_iris_landmarks) == len(GazeParams.RIGHT_IRIS_INDICES)):
            
            left_h_ratio, left_v_ratio, left_gaze_3d, left_pupil = self._calculate_iris_position(
                left_eye_landmarks, left_iris_landmarks, frame)
            
            right_h_ratio, right_v_ratio, right_gaze_3d, right_pupil = self._calculate_iris_position(
                right_eye_landmarks, right_iris_landmarks, frame)
            
            if left_pupil:
                smooth_left_pupil = self._smooth_value(
                    self.left_pupil_center_history, left_pupil)
                self.left_pupil_center_history.append(smooth_left_pupil)
            
            if right_pupil:
                smooth_right_pupil = self._smooth_value(
                    self.right_pupil_center_history, right_pupil)
                self.right_pupil_center_history.append(smooth_right_pupil)

            if left_gaze_3d:
                self.left_gaze_direction_history.append(left_gaze_3d)
            
            if right_gaze_3d:
                self.right_gaze_direction_history.append(right_gaze_3d)
            
            direction, confidence, compensated_h_ratio, compensated_v_ratio = self._determine_gaze_direction(
                left_h_ratio, left_v_ratio, left_gaze_3d,
                right_h_ratio, right_v_ratio, right_gaze_3d,
                head_pitch=head_pitch_rad,
                head_yaw=head_yaw_rad
            )

            gaze_result["direction"] = direction
            gaze_result["confidence"] = float(confidence)
            gaze_result["horizontal_ratio"] = float(compensated_h_ratio)
            gaze_result["vertical_ratio"] = float(compensated_v_ratio)
            gaze_result["left_eye"]["iris_position"] = (float(left_h_ratio), float(left_v_ratio))
            gaze_result["right_eye"]["iris_position"] = (float(right_h_ratio), float(right_v_ratio))
            gaze_result["left_eye"]["eye_landmarks"] = [(float(x), float(y)) for x, y in left_eye_landmarks]
            gaze_result["right_eye"]["eye_landmarks"] = [(float(x), float(y)) for x, y in right_eye_landmarks]
            
            if self.debug:
                for point in left_eye_landmarks:
                    cv2.circle(frame, point, 1, (0, 255, 0), -1)
                for point in right_eye_landmarks:
                    cv2.circle(frame, point, 1, (0, 255, 0), -1)
                for point in left_iris_landmarks:
                    cv2.circle(frame, point, 1, (0, 0, 255), -1)
                for point in right_iris_landmarks:
                    cv2.circle(frame, point, 1, (0, 0, 255), -1)
                
                if self.left_pupil_center_history:
                    left_pupil = self.left_pupil_center_history[-1]
                    cv2.circle(frame, (int(left_pupil[0]), int(left_pupil[1])), 3, (255, 0, 255), -1)
                
                if self.right_pupil_center_history:
                    right_pupil = self.right_pupil_center_history[-1]
                    cv2.circle(frame, (int(right_pupil[0]), int(right_pupil[1])), 3, (255, 0, 255), -1)
        
        return gaze_result
    
    def _process_frame(self, frame: np.ndarray) -> HeadPoseState:
        """
        处理图像帧，检测头部姿态、动作和视线方向
        
        Args:
            frame: 输入图像帧
            
        Returns:
            HeadPoseState: 头部姿态状态
        """
        state = HeadPoseState(frame=frame, timestamp=time.time())
        
        h, w, _ = frame.shape
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)

        current_head_pitch_rad = 0.0
        current_head_yaw_rad = 0.0
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            state.detections["head_pose"]["detected"] = True
            
            landmarks_list = []
            for i, landmark_mp in enumerate(face_landmarks.landmark):
                x, y, z = landmark_mp.x, landmark_mp.y, landmark_mp.z
                landmarks_list.append((x, y, z))
            state.detections["head_pose"]["landmarks"] = landmarks_list
            
            features = self._extract_features(face_landmarks, frame.shape)
            
            if features is not None:
                current_head_pitch_rad = float(features[0])  # pitch
                current_head_yaw_rad = float(features[1])    # yaw
                current_head_roll_rad = float(features[2])   # roll
                
                state.detections["head_pose"]["pitch"] = current_head_pitch_rad
                state.detections["head_pose"]["yaw"] = current_head_yaw_rad
                state.detections["head_pose"]["roll"] = current_head_roll_rad

            gaze_result = self._process_gaze_direction(frame, face_landmarks, h, w, 
                                                       current_head_pitch_rad, current_head_yaw_rad)
            state.detections["gaze_direction"] = gaze_result

            if features is not None:
                expected_feature_count = self.feature_dim // 2
                if len(features) != expected_feature_count:
                    if len(features) > expected_feature_count:
                        features = features[:expected_feature_count]
                    else:
                        padding = np.zeros(expected_feature_count - len(features), dtype=np.float32)
                        features = np.concatenate([features, padding])
                
                self.features_queue.append(features)

                state.detections["head_pose"]["pitch"] = float(features[0])  # 俯仰角
                state.detections["head_pose"]["yaw"] = float(features[1])    # 偏航角
                state.detections["head_pose"]["roll"] = float(features[2])   # 翻滚角
                
                current_time = time.time()
                if current_time - self._last_status_update >= HeadPoseParams.STATUS_UPDATE_INTERVAL:
                    self._last_status_update = current_time

                    pose, confidence = self._predict_head_pose()
                    
                    if pose and confidence > HeadPoseParams.CONFIDENCE_THRESHOLD:
                        self._current_status = pose
                        self._current_status_confidence = confidence
                        self._current_is_nodding = (pose == HeadPoseParams.STATUS_NODDING)
                        self._current_is_shaking = (pose == HeadPoseParams.STATUS_SHAKING)
                
                state.detections["head_movement"]["is_nodding"] = self._current_is_nodding
                state.detections["head_movement"]["is_shaking"] = self._current_is_shaking
                state.detections["head_movement"]["nod_confidence"] = float(self._current_status_confidence) if self._current_is_nodding else 0.0
                state.detections["head_movement"]["shake_confidence"] = float(self._current_status_confidence) if self._current_is_shaking else 0.0
                state.detections["head_movement"]["status"] = self._current_status
                state.detections["head_movement"]["status_confidence"] = float(self._current_status_confidence)
        
        return state
    
    def start(self) -> int:
        """
        开始头部姿态跟踪
        
        Returns:
            int: 错误码，0表示成功，其他值表示失败
        """
        result = super().start()
        if result != SUCCESS:
            logger.error(f"无法启动头部姿态跟踪器: {result}")
            return result
            
        try:
            self._stop_event.clear()
            self._processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
            self._processing_thread.start()
            logger.info("头部姿态跟踪器已开始运行")
            return SUCCESS
        except Exception as e:
            logger.error(f"启动处理线程时出错: {str(e)}")
            return RUNTIME_ERROR
    
    def shutdown(self) -> int:
        """
        关闭头部姿态跟踪器资源
        
        Returns:
            int: 错误码，0表示成功，其他值表示失败
        """
        try:
            if self._processing_thread and self._processing_thread.is_alive():
                logger.info("正在停止处理线程...")
                self._stop_event.set()
                self._processing_thread.join(timeout=2.0) 
                if self._processing_thread.is_alive():
                    logger.warning("处理线程未能正常结束")
                else:
                    logger.info("处理线程已正常结束")
            
            if self.face_mesh:
                self.face_mesh.close()
                self.face_mesh = None
                logger.info("MediaPipe资源已释放")
            
            result = super().shutdown()
            if result == SUCCESS:
                logger.info("头部姿态跟踪器已关闭")
            else:
                logger.error(f"关闭头部姿态跟踪器时出错: {result}")
            
            return result
            
        except Exception as e:
            logger.error(f"关闭头部姿态跟踪器失败: {str(e)}")
            return RUNTIME_ERROR
