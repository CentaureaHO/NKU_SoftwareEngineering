"""
使用GRU深度学习模型的头部姿态跟踪器实现。
基于训练好的模型，可以检测静止、点头、摇头等头部姿势。
"""
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


class HeadPoseTrackerGRU(BaseVisualModality):
    """
    基于GRU深度学习模型的头部姿态跟踪器
    
    使用训练好的GRU模型检测头部姿势和动作，包括静止、点头、摇头等
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
        
        # 面部关键点3D坐标
        self.face_3d_coords = np.array([
            [285, 528, 200],  # 鼻子
            [285, 371, 152],  # 鼻子下方
            [197, 574, 128],  # 左脸边缘
            [173, 425, 108],  # 左眼附近
            [360, 574, 128],  # 右脸边缘
            [391, 425, 108]   # 右眼附近
        ], dtype=np.float64)

        # GRU模型相关
        self.model = None
        self.scaler = None
        self.config = None
        self.window_size = None
        self.stride = None
        self.feature_dim = None
        self.gesture_mapping = None
        self.inverse_gesture_mapping = None
        
        # 特征历史队列
        self.features_queue = deque(maxlen=HeadPoseParams.HISTORY_LEN)
        
        self._processing_thread = None
        self._stop_event = threading.Event()
        
        self._latest_state = HeadPoseState()
        self._state_lock = threading.Lock()
        
        self._last_status_update = time.time()
        
        # 状态缓存
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
    
    def _process_frame(self, frame: np.ndarray) -> HeadPoseState:
        """
        处理图像帧，检测头部姿态和动作
        
        Args:
            frame: 输入图像帧
            
        Returns:
            HeadPoseState: 头部姿态状态
        """
        state = HeadPoseState(frame=frame, timestamp=time.time())
        
        h, w, _ = frame.shape
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            
            # 提取关键点
            landmarks_list = []
            for i, landmark in enumerate(face_landmarks.landmark):
                if i in HeadPoseParams.ESSENTIAL_LANDMARKS:
                    x, y, z = landmark.x, landmark.y, landmark.z
                    landmarks_list.append((x, y, z))
            
            state.detections["head_pose"]["landmarks"] = landmarks_list
            state.detections["head_pose"]["detected"] = True
            
            # 提取特征
            features = self._extract_features(face_landmarks, frame.shape)
            if features is not None:
                # 确保特征维度正确
                expected_feature_count = self.feature_dim // 2
                if len(features) != expected_feature_count:
                    if len(features) > expected_feature_count:
                        features = features[:expected_feature_count]
                    else:
                        padding = np.zeros(expected_feature_count - len(features), dtype=np.float32)
                        features = np.concatenate([features, padding])
                
                # 更新特征队列
                self.features_queue.append(features)
                
                # 提取基本头部姿态信息
                state.detections["head_pose"]["pitch"] = float(features[0])  # pitch
                state.detections["head_pose"]["yaw"] = float(features[1])    # yaw
                state.detections["head_pose"]["roll"] = float(features[2])   # roll
                
                # 预测头部姿势
                current_time = time.time()
                if current_time - self._last_status_update >= HeadPoseParams.STATUS_UPDATE_INTERVAL:
                    self._last_status_update = current_time
                    
                    # 使用GRU模型进行预测
                    pose, confidence = self._predict_head_pose()
                    
                    if pose and confidence > HeadPoseParams.CONFIDENCE_THRESHOLD:
                        self._current_status = pose
                        self._current_status_confidence = confidence
                        self._current_is_nodding = (pose == HeadPoseParams.STATUS_NODDING)
                        self._current_is_shaking = (pose == HeadPoseParams.STATUS_SHAKING)
                
                # 更新状态
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
        
    def get_key_info(self) -> str:
        """
        获取模态的关键信息

        Returns:
            str: 模态的关键信息
        """
        key_info = None
        state = self.update()
        if state.detections['head_movement']['is_nodding']:
            key_info = "点头"
        elif state.detections['head_movement']['is_shaking']:
            key_info = "摇头"
        print(f"key_info: {key_info}")
        return key_info