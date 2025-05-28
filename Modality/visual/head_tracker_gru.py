"""
头部姿态跟踪模块 - 使用GRU深度学习模型

该模块提供了基于GRU模型的头部姿态跟踪功能，能够检测头部的俯仰、偏航和翻滚角度，
以及识别点头和摇头等动作。模块使用MediaPipe人脸网格作为底层人脸特征提取器。
"""

import json
import logging
import os
import pickle
import sys
import threading
import time
from collections import deque, namedtuple
from typing import Tuple, Dict, Any, List, Optional

import cv2
import mediapipe as mp
import numpy as np
import tensorflow

from modality.core.error_codes import (MEDIAPIPE_INITIALIZATION_FAILED,
                                       MODEL_LOADING_FAILED, RUNTIME_ERROR,
                                       SUCCESS)
from modality.visual.base_visual import BaseVisualModality
from modality.visual.head_pose_common import (HeadPoseParams, HeadPoseState,
                                              euclidean_dist,
                                              rotation_matrix_to_angles)

logging.basicConfig(
    level=logging.DEBUG if os.environ.get(
        'MODALITY_DEBUG', '0') == '1' else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='./database/log/head_tracker_gru.log',
    filemode='w',
    encoding='utf-8'
)
logger = logging.getLogger('HeadTrackerGRU')

DEBUG = os.environ.get('MODALITY_DEBUG', '0') == '1'

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 修复 tensorflow.keras 导入
keras = tensorflow.keras

# 创建命名元组用于特征计算
FeatureParams = namedtuple('FeatureParams', [
    'angles', 'distances', 'face_box', 'ratios'
])


class HeadPoseTrackerGRU(BaseVisualModality):
    """
    基于GRU深度学习模型的头部姿态跟踪器，集成了视线方向检测功能

    使用训练好的GRU模型检测头部姿势和动作，包括静止、点头、摇头等，
    同时追踪视线方向
    """

    def __init__(
            self,
            name: str = "head_pose_tracker_gru",
            source: int = 0,
            width: int = 640,
            height: int = 480,
            **kwargs
    ):
        """
        初始化基于GRU模型的头部姿态跟踪器

        Args:
            name: 模态名称
            source: 视频源，可以是摄像头ID或视频文件路径
            width: 图像宽度
            height: 图像高度
            **kwargs: 其他配置参数，包括:
                min_detection_confidence: 检测置信度阈值 (默认0.5)
                min_tracking_confidence: 跟踪置信度阈值 (默认0.5)
                model_dir: 模型目录路径 (默认None)
                debug: 是否启用调试模式 (默认DEBUG)
        """
        super().__init__(name, source, width, height)

        # 从kwargs获取配置参数，提供默认值
        self._config = {
            'min_detection_confidence': kwargs.get('min_detection_confidence', 0.5),
            'min_tracking_confidence': kwargs.get('min_tracking_confidence', 0.5),
            'debug': kwargs.get('debug', DEBUG),
            'model_dir': kwargs.get('model_dir', None)
        }

        # 设置模型目录
        if self._config['model_dir'] is None:
            self._config['model_dir'] = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "models", "headpose"
            )

        # 合并多个属性到更少的字典中，减少实例属性
        self._runtime = {
            # 模型相关属性
            'model': None,
            'scaler': None,
            'config': None,
            'window_size': None,
            'stride': None,
            'feature_dim': None,
            'gesture_mapping': None,
            'inverse_gesture_mapping': None,
            'features_queue': deque(maxlen=HeadPoseParams.HISTORY_LEN),
            # 状态跟踪属性
            'last_status_update': time.time(),
            'current_is_nodding': False,
            'current_is_shaking': False,
            'current_status': HeadPoseParams.STATUS_STATIONARY,
            'current_status_confidence': 0.0,
            'h_ratio_variance': 0.1,
            'v_ratio_variance': 0.1,
            # 线程相关属性
            'processing_thread': None,
            'stop_event': threading.Event(),
            'latest_state': HeadPoseState(),
            'state_lock': threading.Lock()
        }

        # MediaPipe设置
        self.face_mesh = None
        self.mp_face_options = {
            'max_num_faces': 1,
            'refine_landmarks': True,
            'min_detection_confidence': self._config['min_detection_confidence'],
            'min_tracking_confidence': self._config['min_tracking_confidence'],
        }

        # 3D坐标模型
        self.face_3d_coords = np.array([
            [285, 528, 200],  # 鼻子
            [285, 371, 152],  # 鼻子下方
            [197, 574, 128],  # 左脸边缘
            [173, 425, 108],  # 左眼附近
            [360, 574, 128],  # 右脸边缘
            [391, 425, 108]   # 右眼附近
        ], dtype=np.float64)

    @property
    def model(self):
        """获取模型实例"""
        return self._runtime['model']

    @property
    def scaler(self):
        """获取特征缩放器"""
        return self._runtime['scaler']

    @property
    def config(self):
        """获取模型配置"""
        return self._runtime['config']

    @property
    def window_size(self):
        """获取窗口大小"""
        return self._runtime['window_size']

    @property
    def feature_dim(self):
        """获取特征维度"""
        return self._runtime['feature_dim']

    @property
    def features_queue(self):
        """获取特征队列"""
        return self._runtime['features_queue']

    def initialize(self) -> int:
        """
        初始化摄像头、MediaPipe人脸网格和GRU模型

        Returns:
            int: 错误码，0表示成功，其他值表示失败
        """
        # 调用基类初始化
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
            return self._load_model()
        except (ImportError, IOError) as e:
            logger.error("资源导入或IO错误: %s", str(e))
            return MEDIAPIPE_INITIALIZATION_FAILED
        except (ValueError, RuntimeError) as e:
            logger.error("初始化参数或运行时错误: %s", str(e))
            return MEDIAPIPE_INITIALIZATION_FAILED

    def _load_model_file(self, model_path: str) -> Optional[Any]:
        """加载模型文件"""
        try:
            logger.info("正在加载模型: %s", model_path)
            return keras.models.load_model(model_path)
        except (ImportError, IOError) as e:
            logger.error("加载模型出错: %s", str(e))
            return None

    def _load_scaler_file(self, scaler_path: str) -> Optional[Any]:
        """加载缩放器文件"""
        try:
            logger.info("正在加载缩放器: %s", scaler_path)
            with open(scaler_path, 'rb') as f:
                return pickle.load(f)
        except (pickle.UnpicklingError, IOError) as e:
            logger.error("加载缩放器出错: %s", str(e))
            return None

    def _load_config_file(self, config_path: str) -> Optional[Dict]:
        """加载配置文件"""
        try:
            logger.info("正在加载配置: %s", config_path)
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.error("加载配置出错: %s", str(e))
            return None

    def _load_model(self) -> int:
        """
        加载GRU模型、缩放器和配置

        Returns:
            int: 错误码，0表示成功，其他值表示失败
        """
        # 找到模型文件
        model_path = os.path.join(self._config['model_dir'], "model.h5")
        scaler_path = os.path.join(self._config['model_dir'], "scaler.pkl")
        config_path = os.path.join(self._config['model_dir'], "config.json")

        # 检查所有文件是否存在
        for path, name in [(model_path, "模型"), (scaler_path, "缩放器"), (config_path, "配置")]:
            if not os.path.exists(path):
                logger.error(f"{name}文件不存在: %s", path)
                return MODEL_LOADING_FAILED

        # 尝试加载所有必需的组件
        load_results = {
            'model': self._load_model_file(model_path),
            'scaler': self._load_scaler_file(scaler_path),
            'config': self._load_config_file(config_path)
        }

        # 检查是否所有组件都成功加载
        for component, result in load_results.items():
            if result is None:
                logger.error("加载%s失败", component)
                return MODEL_LOADING_FAILED
            self._runtime[component] = result

        # 设置模型参数
        config = self._runtime['config']
        self._runtime['window_size'] = config.get(
            "window_size", HeadPoseParams.HISTORY_LEN)
        self._runtime['stride'] = config.get("stride", 5)
        self._runtime['feature_dim'] = config.get("feature_dim", 30)

        # 设置姿势映射
        gesture_mapping = config.get("gesture_mapping", {
            "stationary": 0, "nodding": 1, "shaking": 2, "other": 3
        })
        self._runtime['gesture_mapping'] = gesture_mapping
        self._runtime['inverse_gesture_mapping'] = {
            v: k for k, v in gesture_mapping.items()}

        # 初始化特征队列
        self._runtime['features_queue'] = deque(
            maxlen=self._runtime['window_size'])

        logger.info(
            "模型加载成功，窗口大小: %d, 特征维度: %d",
            self._runtime['window_size'],
            self._runtime['feature_dim']
        )
        logger.info("姿势映射: %s", self._runtime['gesture_mapping'])

        return SUCCESS

    def _processing_loop(self):
        """处理线程的主循环，负责连续处理视频帧"""
        logger.info("处理线程已启动")

        frame_interval = 1.0 / HeadPoseParams.VIDEO_FPS
        while not self._runtime['stop_event'].is_set():
            start_time = time.time()

            # 检查相机是否在运行
            if not self.camera_manager.is_running():
                logger.warning(
                    "CameraManager is not running. Trying to initialize...")
                # 尝试初始化
                init_status = self.camera_manager.initialize_camera(
                    self.source, self.width, self.height, self.loop_video
                )
                if init_status != SUCCESS:
                    logger.error(
                        "Failed to re-initialize camera in processing loop. Error: %d",
                        init_status
                    )
                    time.sleep(1)
                    continue

            # 读取帧
            ret, frame = self.camera_manager.read_frame()

            if not ret:
                # 处理视频结束或相机错误情况
                if (self.camera_manager.config.is_file_source and
                        not self.camera_manager.config.loop_video):
                    logger.info(
                        "End of non-looping video file reached in HeadPoseTrackerGRU.")
                    break
                logger.warning(
                    "Failed to get frame from CameraManager in _processing_loop. Retrying..."
                )
                time.sleep(0.1)
                continue

            # 处理帧
            state = self._process_frame(frame)
            with self._runtime['state_lock']:
                self._runtime['latest_state'] = state

            # 控制帧率
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
        processing_thread = self._runtime['processing_thread']

        if self._is_running and processing_thread and processing_thread.is_alive():
            with self._runtime['state_lock']:
                return self._runtime['latest_state']
        elif not self._is_running:
            logger.warning(
                "%s modality is not running. Returning empty state.", self.name)
            return HeadPoseState()
        else:
            # 尝试同步读取帧
            logger.warning(
                "Processing thread is not alive. Attempting a synchronous frame read."
            )
            if not self.camera_manager.is_running():
                logger.error(
                    "Camera manager not running for synchronous update.")
                return HeadPoseState()

            ret, frame = self.camera_manager.read_frame()
            if not ret or frame is None:
                logger.error("Failed to read frame synchronously.")
                return HeadPoseState()

            state = self._process_frame(frame)
            with self._runtime['state_lock']:
                self._runtime['latest_state'] = state
            return state

    def _collect_face_landmarks(self, face_landmarks, frame_shape) -> Tuple[List, Dict, Dict, List]:
        """
        收集人脸关键点数据

        Args:
            face_landmarks: MediaPipe人脸网格关键点
            frame_shape: 图像帧形状

        Returns:
            tuple: 关键点和人脸框信息
        """
        h, w = frame_shape[:2]

        # 收集关键点
        face_coordination_in_image = []
        key_points = {
            'nose_point': None,
            'chin_point': None,
            'left_ear_point': None,
            'left_face_point': None,
            'right_ear_point': None,
            'right_face_point': None
        }
        all_face_coords = []

        for idx, lm in enumerate(face_landmarks.landmark):
            x, y = int(lm.x * w), int(lm.y * h)
            all_face_coords.append((x, y))

            if idx in [1, 9, 57, 130, 287, 359]:
                face_coordination_in_image.append([x, y])

            # 收集特定关键点
            if idx == HeadPoseParams.LANDMARK_NOSE:  # 鼻尖
                key_points['nose_point'] = (x, y)
            elif idx == HeadPoseParams.LANDMARK_CHIN:  # 下巴
                key_points['chin_point'] = (x, y)
            elif idx == HeadPoseParams.LANDMARK_LEFT_EAR:  # 左耳
                key_points['left_ear_point'] = (x, y)
            elif idx == HeadPoseParams.LANDMARK_LEFT_FACE:  # 左脸中心点
                key_points['left_face_point'] = (x, y)
            elif idx == HeadPoseParams.LANDMARK_RIGHT_EAR:  # 右耳
                key_points['right_ear_point'] = (x, y)
            elif idx == HeadPoseParams.LANDMARK_RIGHT_FACE:  # 右脸中心点
                key_points['right_face_point'] = (x, y)

        # 计算人脸框信息
        all_face_coords_np = np.array(all_face_coords)
        face_box_info = self._compute_face_box(all_face_coords_np)

        return (
            face_coordination_in_image,
            key_points,
            face_box_info,
            all_face_coords
        )

    def _compute_face_box(self, all_face_coords: np.ndarray) -> Dict[str, float]:
        """计算人脸框信息"""
        x_min, y_min = np.min(all_face_coords, axis=0)
        x_max, y_max = np.max(all_face_coords, axis=0)
        box_width = max(x_max - x_min, 1)
        box_height = max(y_max - y_min, 1)
        box_diagonal = np.sqrt(box_width**2 + box_height**2)
        aspect_ratio = box_width / box_height

        return {
            'box_width': box_width,
            'box_height': box_height,
            'box_diagonal': box_diagonal,
            'aspect_ratio': aspect_ratio
        }

    def _compute_head_pose_features(self, face_data: Dict) -> Optional[np.ndarray]:
        """计算头部姿态特征

        Args:
            face_data: 包含人脸坐标和尺寸信息的字典

        Returns:
            np.ndarray: 提取的特征向量或None
        """
        try:
            # 解包数据
            face_coords = face_data['coords']
            key_points = face_data['key_points']
            face_box_info = face_data['box_info']

            # 计算相机矩阵
            cam_info = self._calculate_camera_matrix(
                face_data['width'], face_data['height'])

            # 计算旋转向量和角度
            angles = self._calculate_pose_angles(face_coords, cam_info)
            if angles is None:
                return None

            # 提取关键尺寸数据
            box_info = {
                'width': face_box_info['box_width'],
                'height': face_box_info['box_height'],
                'diagonal': face_box_info['box_diagonal'],
                'aspect_ratio': face_box_info['aspect_ratio']
            }

            # 计算几何特征
            distances = self._calculate_distances(key_points)

            # 组合特征参数
            params = FeatureParams(
                angles=angles,
                distances=distances,
                face_box=box_info,
                ratios={
                    'nose_chin_ratio': distances['nose_chin'] / box_info['height'],
                    'left_cheek_ratio': distances['left_cheek'] / box_info['width'],
                    'right_cheek_ratio': distances['right_cheek'] / box_info['width']
                }
            )

            # 创建特征向量
            return self._create_feature_vector(params)

        except (ValueError, IndexError) as e:
            logger.warning("计算头部姿态特征时出错: %s", str(e))
            return None
        except (AttributeError, TypeError) as e:
            logger.warning("特征类型错误: %s", str(e))
            return None

    def _calculate_camera_matrix(self, width: int, height: int) -> Dict:
        """计算相机内参矩阵"""
        focal_length = 1 * width
        cam_matrix = np.array([
            [focal_length, 0, width / 2],
            [0, focal_length, height / 2],
            [0, 0, 1]
        ])
        dist_matrix = np.zeros((4, 1), dtype=np.float64)
        return {'cam_matrix': cam_matrix, 'dist_matrix': dist_matrix}

    def _calculate_pose_angles(
        self,
        face_coords: np.ndarray,
        cam_info: Dict
    ) -> Optional[List[float]]:
        """计算头部姿态角度"""
        success, rotation_vec, _ = cv2.solvePnP(
            self.face_3d_coords, face_coords,
            cam_info['cam_matrix'], cam_info['dist_matrix']
        )

        if not success:
            return None

        rotation_matrix, _ = cv2.Rodrigues(rotation_vec)
        angles = rotation_matrix_to_angles(rotation_matrix)
        return [float(a) for a in angles]  # pitch, yaw, roll

    def _calculate_distances(self, key_points: Dict) -> Dict:
        """计算关键点之间的距离"""
        return {
            'nose_chin': euclidean_dist(
                key_points['nose_point'], key_points['chin_point']
            ),
            'left_cheek': euclidean_dist(
                key_points['left_ear_point'], key_points['left_face_point']
            ),
            'right_cheek': euclidean_dist(
                key_points['right_ear_point'], key_points['right_face_point']
            )
        }

    def _create_feature_vector(self, params: FeatureParams) -> np.ndarray:
        """创建特征向量"""
        pitch, yaw, roll = params.angles
        box = params.face_box
        dist = params.distances
        ratios = params.ratios

        basic_features = np.array([
            pitch, yaw, roll,
            dist['nose_chin'], dist['left_cheek'], dist['right_cheek']
        ], dtype=np.float32)

        normalized_features = np.array([
            box['aspect_ratio'],
            box['width'] / box['diagonal'],
            box['height'] / box['diagonal'],
            pitch / box['diagonal'],
            yaw / box['diagonal'],
            roll / box['diagonal'],
            ratios['nose_chin_ratio'],
            ratios['left_cheek_ratio'],
            ratios['right_cheek_ratio']
        ], dtype=np.float32)

        return np.concatenate([basic_features, normalized_features])

    def _extract_features(self, face_landmarks, frame_shape) -> Optional[np.ndarray]:
        """
        从人脸关键点提取特征

        Args:
            face_landmarks: MediaPipe人脸网格关键点
            frame_shape: 图像帧形状

        Returns:
            np.ndarray: 提取的特征向量或None
        """
        h, w = frame_shape[:2]

        try:
            face_coords, key_points, face_box_info, _ = self._collect_face_landmarks(
                face_landmarks, frame_shape
            )
        except (ValueError, IndexError) as e:
            logger.warning("提取关键点时出错: %s", str(e))
            return None

        if None in key_points.values():
            return None

        if len(face_coords) != 6:
            return None

        face_data = {
            'coords': np.array(face_coords, dtype=np.float64),
            'key_points': key_points,
            'box_info': face_box_info,
            'width': w,
            'height': h
        }

        return self._compute_head_pose_features(face_data)

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
            pose_name = self._runtime['inverse_gesture_mapping'].get(
                gesture_idx, HeadPoseParams.STATUS_STATIONARY)

            return pose_name, confidence

        except (ValueError, IndexError) as e:
            logger.error("预测头部姿势时出错: %s", str(e))
            return HeadPoseParams.STATUS_STATIONARY, 0.0

    def _process_landmarks(self, face_landmarks, frame):
        """处理人脸关键点"""
        state = HeadPoseState(frame=frame, timestamp=time.time())
        state.detections["head_pose"]["detected"] = True

        # 提取所有关键点
        landmarks_list = []
        for _, landmark_mp in enumerate(face_landmarks.landmark):
            x, y, z = landmark_mp.x, landmark_mp.y, landmark_mp.z
            landmarks_list.append((x, y, z))
        state.detections["head_pose"]["landmarks"] = landmarks_list

        # 提取特征
        features = self._extract_features(face_landmarks, frame.shape)

        # 处理提取的特征
        if features is None:
            return state

        # 基本角度信息
        state.detections["head_pose"]["pitch"] = float(features[0])  # pitch
        state.detections["head_pose"]["yaw"] = float(features[1])    # yaw
        state.detections["head_pose"]["roll"] = float(features[2])   # roll

        # 处理特征向量，确保长度正确
        expected_feature_count = self.feature_dim // 2
        if len(features) != expected_feature_count:
            features = self._adjust_feature_length(
                features, expected_feature_count)

        # 添加到特征队列
        self.features_queue.append(features)

        # 定期更新状态
        self._update_head_movement_status(state)

        return state

    def _adjust_feature_length(self, features: np.ndarray, expected_length: int) -> np.ndarray:
        """调整特征向量长度"""
        if len(features) > expected_length:
            return features[:expected_length]
        padding = np.zeros(expected_length -
                           len(features), dtype=np.float32)
        return np.concatenate([features, padding])

    def _update_head_movement_status(self, state: HeadPoseState):
        """更新头部运动状态"""
        current_time = time.time()
        if (current_time - self._runtime['last_status_update'] >=
                HeadPoseParams.STATUS_UPDATE_INTERVAL):
            self._runtime['last_status_update'] = current_time

            # 预测头部姿势
            pose, confidence = self._predict_head_pose()

            # 更新状态
            if pose and confidence > HeadPoseParams.CONFIDENCE_THRESHOLD:
                self._runtime['current_status'] = pose
                self._runtime['current_status_confidence'] = confidence
                self._runtime['current_is_nodding'] = (
                    pose == HeadPoseParams.STATUS_NODDING)
                self._runtime['current_is_shaking'] = (
                    pose == HeadPoseParams.STATUS_SHAKING)

        # 填充检测结果
        state.detections["head_movement"]["is_nodding"] = (
            self._runtime['current_is_nodding'])
        state.detections["head_movement"]["is_shaking"] = (
            self._runtime['current_is_shaking'])
        state.detections["head_movement"]["nod_confidence"] = float(
            self._runtime['current_status_confidence']
            if self._runtime['current_is_nodding'] else 0.0)
        state.detections["head_movement"]["shake_confidence"] = float(
            self._runtime['current_status_confidence']
            if self._runtime['current_is_shaking'] else 0.0)
        state.detections["head_movement"]["status"] = (
            self._runtime['current_status'])
        state.detections["head_movement"]["status_confidence"] = float(
            self._runtime['current_status_confidence'])

    def _process_frame(self, frame: np.ndarray) -> HeadPoseState:
        """
        处理图像帧，检测头部姿态、动作和视线方向

        Args:
            frame: 输入图像帧

        Returns:
            HeadPoseState: 头部姿态状态
        """
        state = HeadPoseState(frame=frame, timestamp=time.time())

        # 转换为RGB并进行检测
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)

        # 检测到人脸
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            return self._process_landmarks(face_landmarks, frame)

        return state

    def start(self) -> int:
        """
        开始头部姿态跟踪

        Returns:
            int: 错误码，0表示成功，其他值表示失败
        """
        # 确保相机已初始化
        if not self.camera_manager.is_running():
            logger.info(
                "Camera for source %s not running. Initializing before start...",
                self.source
            )
            init_status = self.camera_manager.initialize_camera(
                source=self.source,
                width=self.width,
                height=self.height,
                loop_video=self.loop_video
            )
            if init_status != SUCCESS:
                logger.error(
                    "Failed to initialize camera via CameraManager before starting. Error: %d",
                    init_status
                )
                return init_status

        # 启动基类
        result = super().start()
        if result != SUCCESS:
            logger.error("BaseModality start failed for %s: %d",
                         self.name, result)
            return result

        try:
            # 启动处理线程
            self._runtime['stop_event'].clear()
            self._runtime['processing_thread'] = threading.Thread(
                target=self._processing_loop, daemon=True)
            self._runtime['processing_thread'].start()
            logger.info("%s has started processing.", self.name)
            return SUCCESS
        except RuntimeError as e:
            logger.error("Error starting processing thread for %s: %s",
                         self.name, str(e), exc_info=True)
            super().stop()
            return RUNTIME_ERROR

    def shutdown(self) -> int:
        """
        关闭头部姿态跟踪器资源

        Returns:
            int: 错误码，0表示成功，其他值表示失败
        """
        logger.info("Shutting down %s...", self.name)

        # 停止处理线程
        processing_thread = self._runtime['processing_thread']
        if processing_thread and processing_thread.is_alive():
            logger.info("Stopping processing thread...")
            self._runtime['stop_event'].set()
            processing_thread.join(timeout=2.0)
            if processing_thread.is_alive():
                logger.warning(
                    "Processing thread did not terminate gracefully.")
            else:
                logger.info("Processing thread stopped.")

        # 释放MediaPipe资源
        if self.face_mesh:
            try:
                self.face_mesh.close()
                self.face_mesh = None
                logger.info("MediaPipe FaceMesh resources released.")
            except AttributeError as e:
                logger.error("Error closing MediaPipe FaceMesh: %s",
                             str(e), exc_info=True)

        # 调用基类关闭方法
        result = super().shutdown()
        if result == SUCCESS:
            logger.info("%s shutdown successfully.", self.name)
        else:
            logger.error("Error during super().shutdown() for %s: %d",
                         self.name, result)

        return result

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
        return key_info
