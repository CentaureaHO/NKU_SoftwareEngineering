"""
基于机器学习模型的增强视线方向跟踪模块。
该模块使用预训练的机器学习模型对用户视线方向进行精确分类。
"""
import logging
import os
import pickle
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Optional

import cv2
import mediapipe as mp
import numpy as np

from modality.core.error_codes import (MEDIAPIPE_INITIALIZATION_FAILED,
                                       MODEL_LOADING_FAILED,
                                       RUNTIME_ERROR, SUCCESS)
from modality.visual.base_visual import BaseVisualModality, VisualState

logging.basicConfig(
    level=logging.DEBUG if os.environ.get(
        'MODALITY_DEBUG', '0') == '1' else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='./database/log/gaze_boost.log',
    filemode='w',
    encoding='utf-8'
)
logger = logging.getLogger('GazeBoost')

DEBUG = os.environ.get('MODALITY_DEBUG', '0') == '1'

# MediaPipe 资源
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# 视线方向常量
DIRECTION_CENTER = "center"
DIRECTION_LEFT = "left"
DIRECTION_RIGHT = "right"
DIRECTION_UP = "up"
DIRECTION_DOWN = "down"

# 定义模型路径
MODEL_DIR = os.path.join(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))), "models", "gaze")
HORIZONTAL_MODEL_PATH = os.path.join(MODEL_DIR, "horizontal.pkl")
VERTICAL_MODEL_PATH = os.path.join(MODEL_DIR, "vertical.pkl")

# 关键点索引
EYE_LANDMARK_INDICES = [
    # 左眼轮廓
    362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398,
    # 右眼轮廓
    33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
    # 左眼虹膜
    468, 469, 470, 471, 472,
    # 右眼虹膜
    473, 474, 475, 476, 477
]

# 鼻子、面部姿态相关的关键点
POSE_LANDMARK_INDICES = [1, 33, 263, 61, 291, 199]

# 视频参数
VIDEO_FPS = 30


@dataclass
class EyeData:
    """眼睛关键点数据"""
    eye_landmarks: List[Tuple[int, int]] = field(default_factory=list)
    iris_landmarks: List[Tuple[int, int]] = field(default_factory=list)
    iris_position: Tuple[float, float] = (0.0, 0.0)  # (水平比例, 垂直比例)


@dataclass
class GazeBoostState(VisualState):
    """
    增强视线方向状态类，扩展自VisualState，存储机器学习模型预测的视线方向信息
    """

    def __init__(self, frame=None, timestamp=None):
        super().__init__(frame, timestamp)
        self.detections = {
            "gaze_direction": {
                "horizontal": {
                    # 水平视线方向(left, center, right)
                    "direction": DIRECTION_CENTER,
                    "confidence": 0.0                # 置信度
                },
                "vertical": {
                    "direction": DIRECTION_CENTER,   # 垂直视线方向(up, center, down)
                    "confidence": 0.0                # 置信度
                },
                "combined_direction": DIRECTION_CENTER,  # 组合方向
                "iris_position": {
                    "horizontal_ratio": 0.0,         # 水平比例(-1到1，负为左，正为右)
                    "vertical_ratio": 0.0,           # 垂直比例(-1到1，负为上，正为下)
                    "left_eye": (0.0, 0.0),          # 左眼虹膜位置比例
                    "right_eye": (0.0, 0.0)          # 右眼虹膜位置比例
                },
                "face_detected": False               # 是否检测到人脸
            }
        }

    def to_dict(self) -> Dict[str, Any]:
        """将状态转换为字典格式"""
        result = super().to_dict()
        result.update({"gaze boost": self.detections["gaze_direction"]})
        return result

    def is_looking_center(self) -> bool:
        """检查用户是否注视中央"""
        h_dir = self.detections["gaze_direction"]["horizontal"]["direction"]
        v_dir = self.detections["gaze_direction"]["vertical"]["direction"]
        return h_dir == DIRECTION_CENTER and v_dir == DIRECTION_CENTER

    def get_horizontal_direction(self) -> str:
        """获取当前水平视线方向"""
        return self.detections["gaze_direction"]["horizontal"]["direction"]

    def get_vertical_direction(self) -> str:
        """获取当前垂直视线方向"""
        return self.detections["gaze_direction"]["vertical"]["direction"]

    def get_combined_direction(self) -> str:
        """获取组合视线方向"""
        return self.detections["gaze_direction"]["combined_direction"]


class GazeBoost(BaseVisualModality):
    """
    基于机器学习模型的增强视线方向跟踪器，提供更精确的视线方向分类
    """

    def __init__(self,
                 name: str = "gaze_boost",
                 source: int = 0,
                 width: int = 640,
                 height: int = 480,
                 **kwargs):
        """
        初始化增强视线方向跟踪器

        Args:
            name: 模态名称
            source: 视频源，可以是摄像头ID或视频文件路径
            width: 图像宽度
            height: 图像高度
            kwargs: 其他可选参数，可包含:
                min_detection_confidence: 检测置信度阈值
                min_tracking_confidence: 跟踪置信度阈值
                debug: 是否启用调试模式
        """
        super().__init__(name, source, width, height)

        # 提取可选参数
        min_detection_confidence = kwargs.get('min_detection_confidence', 0.5)
        min_tracking_confidence = kwargs.get('min_tracking_confidence', 0.5)
        self.debug = kwargs.get('debug', DEBUG)

        self.mp_face_options = {
            'max_num_faces': 1,
            'refine_landmarks': True,
            'min_detection_confidence': min_detection_confidence,
            'min_tracking_confidence': min_tracking_confidence,
        }

        # 状态和线程控制
        self._latest_state = GazeBoostState()
        self._state_lock = threading.Lock()
        self._processing_thread = None
        self._stop_event = threading.Event()

        # MediaPipe 和模型
        self.face_mesh = None
        self.horizontal_model = None
        self.vertical_model = None

        # 方向映射常量
        self.HORIZONTAL_LABELS = ["left", "center", "right"]
        self.VERTICAL_LABELS = ["up", "center", "down"]

        # 方向颜色映射
        self.DIRECTION_COLORS = {
            "left": (0, 0, 255),    # 红色
            "center": (0, 255, 0),  # 绿色
            "right": (255, 0, 0),   # 蓝色
            "up": (255, 255, 0),    # 青色
            "down": (255, 0, 255)   # 紫色
        }

    def initialize(self) -> int:
        """
        初始化摄像头、MediaPipe人脸网格和预训练模型

        Returns:
            int: 错误码，0表示成功，其他值表示失败
        """
        # 初始化基础视觉模态
        result = super().initialize()
        if result != SUCCESS:
            logger.error("基础视觉模态初始化失败")
            return result

        try:
            # 初始化MediaPipe
            self.face_mesh = mp_face_mesh.FaceMesh(
                max_num_faces=self.mp_face_options['max_num_faces'],
                refine_landmarks=self.mp_face_options['refine_landmarks'],
                min_detection_confidence=self.mp_face_options['min_detection_confidence'],
                min_tracking_confidence=self.mp_face_options['min_tracking_confidence'],
            )
            logger.info("MediaPipe人脸网格初始化成功")

            # 加载预训练模型
            self.horizontal_model = self._load_model(HORIZONTAL_MODEL_PATH)
            self.vertical_model = self._load_model(VERTICAL_MODEL_PATH)
            logger.info("预训练模型加载成功")

            return SUCCESS
        except (ValueError, RuntimeError, ImportError) as e:
            logger.error("MediaPipe初始化失败: %s", str(e), exc_info=True)
            return MEDIAPIPE_INITIALIZATION_FAILED
        except FileNotFoundError as e:
            logger.error("模型加载失败: %s", str(e), exc_info=True)
            return MODEL_LOADING_FAILED

    def _load_model(self, model_path: str):
        """
        加载预训练模型

        Args:
            model_path: 模型文件路径

        Returns:
            模型对象

        Raises:
            FileNotFoundError: 如果模型文件不存在
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件未找到: {model_path}")

        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        logger.info(f"成功加载模型: {model_path}")
        return model

    def _calculate_iris_position(self, eye_landmarks, iris_landmarks):
        """
        计算虹膜相对于眼睛的位置

        Args:
            eye_landmarks: 眼睛轮廓关键点
            iris_landmarks: 虹膜关键点

        Returns:
            Tuple[float, float]: 水平和垂直位置比例 (-1到1)
        """
        eye_x_coords = [p[0] for p in eye_landmarks]
        eye_y_coords = [p[1] for p in eye_landmarks]

        eye_left = min(eye_x_coords)
        eye_right = max(eye_x_coords)
        eye_top = min(eye_y_coords)
        eye_bottom = max(eye_y_coords)

        eye_width = max(eye_right - eye_left, 1e-5)
        eye_height = max(eye_bottom - eye_top, 1e-5)

        iris_center_x = sum(p[0] for p in iris_landmarks) / len(iris_landmarks)
        iris_center_y = sum(p[1] for p in iris_landmarks) / len(iris_landmarks)

        horizontal_ratio = 2 * (iris_center_x - eye_left) / eye_width - 1
        vertical_ratio = 2 * (iris_center_y - eye_top) / eye_height - 1

        return horizontal_ratio, vertical_ratio

    def _extract_features(self, landmarks, frame_shape):
        """
        从面部关键点提取特征向量

        Args:
            landmarks: MediaPipe面部特征点
            frame_shape: 图像帧的形状

        Returns:
            Tuple: 特征向量、平均水平比例和平均垂直比例
        """
        feature_vector = []

        # 收集必要的关键点
        face_landmarks = {}
        h, w = frame_shape[:2]

        for idx, landmark in enumerate(landmarks.landmark):
            x, y, z = landmark.x, landmark.y, landmark.z
            pixel_x, pixel_y = int(x * w), int(y * h)
            face_landmarks[idx] = {
                'x': x, 'y': y, 'z': z,
                'pixel_x': pixel_x, 'pixel_y': pixel_y
            }

        # 左眼、右眼和虹膜的关键点
        left_eye_landmarks = []
        right_eye_landmarks = []
        left_iris_landmarks = []
        right_iris_landmarks = []

        for idx in range(478):  # MediaPipe Face Mesh有478个关键点
            if idx in face_landmarks:
                point = face_landmarks[idx]
                pixel_point = (point['pixel_x'], point['pixel_y'])

                if idx in EYE_LANDMARK_INDICES[:16]:  # 左眼
                    left_eye_landmarks.append(pixel_point)
                if idx in EYE_LANDMARK_INDICES[16:32]:  # 右眼
                    right_eye_landmarks.append(pixel_point)
                if idx in EYE_LANDMARK_INDICES[32:37]:  # 左虹膜
                    left_iris_landmarks.append(pixel_point)
                if idx in EYE_LANDMARK_INDICES[37:42]:  # 右虹膜
                    right_iris_landmarks.append(pixel_point)

        # 计算虹膜位置
        left_h_ratio, left_v_ratio = 0, 0
        right_h_ratio, right_v_ratio = 0, 0

        if len(left_eye_landmarks) == 16 and len(left_iris_landmarks) == 5:
            left_h_ratio, left_v_ratio = self._calculate_iris_position(
                left_eye_landmarks, left_iris_landmarks)

        if len(right_eye_landmarks) == 16 and len(right_iris_landmarks) == 5:
            right_h_ratio, right_v_ratio = self._calculate_iris_position(
                right_eye_landmarks, right_iris_landmarks)

        avg_h_ratio = (left_h_ratio + right_h_ratio) / 2
        avg_v_ratio = (left_v_ratio + right_v_ratio) / 2

        # 提取特征向量 (与训练时相同格式)
        for idx in EYE_LANDMARK_INDICES + POSE_LANDMARK_INDICES:
            if idx in face_landmarks:
                point = face_landmarks[idx]
                feature_vector.extend([point['x'], point['y'], point['z']])
            else:
                feature_vector.extend([0, 0, 0])

        # 添加虹膜位置特征
        feature_vector.append(left_h_ratio)
        feature_vector.append(left_v_ratio)
        feature_vector.append(right_h_ratio)
        feature_vector.append(right_v_ratio)
        feature_vector.append(avg_h_ratio)
        feature_vector.append(avg_v_ratio)

        return np.array([feature_vector]), avg_h_ratio, avg_v_ratio, (left_h_ratio, left_v_ratio), (right_h_ratio, right_v_ratio)

    def _predict_gaze(self, features):
        """
        使用预训练模型预测视线方向

        Args:
            features: 特征向量

        Returns:
            Tuple: 水平方向、垂直方向、水平置信度、垂直置信度
        """
        if self.horizontal_model is None or self.vertical_model is None:
            return "unknown", "unknown", 0.0, 0.0

        # 预测水平方向 - 使用predict，与test.py保持一致
        h_probs = self.horizontal_model.predict(features)[0]
        h_pred_idx = np.argmax(h_probs)
        h_confidence = h_probs[h_pred_idx]

        # 预测垂直方向 - 使用predict，与test.py保持一致
        v_probs = self.vertical_model.predict(features)[0]
        v_pred_idx = np.argmax(v_probs)
        v_confidence = v_probs[v_pred_idx]

        # 获取标签 - 使用模型中保存的标签映射
        if hasattr(self.horizontal_model, 'label_map'):
            horizontal_pred = self.horizontal_model.label_map[h_pred_idx]
        else:
            horizontal_pred = self.HORIZONTAL_LABELS[h_pred_idx]

        if hasattr(self.vertical_model, 'label_map'):
            vertical_pred = self.vertical_model.label_map[v_pred_idx]
        else:
            vertical_pred = self.VERTICAL_LABELS[v_pred_idx]

        return horizontal_pred, vertical_pred, h_confidence, v_confidence

    def _get_combined_direction(self, horizontal, vertical):
        """
        获取组合视线方向

        Args:
            horizontal: 水平方向
            vertical: 垂直方向

        Returns:
            str: 组合后的方向
        """
        if horizontal == DIRECTION_CENTER and vertical == DIRECTION_CENTER:
            return DIRECTION_CENTER
        elif vertical == DIRECTION_UP and horizontal == DIRECTION_CENTER:
            return DIRECTION_UP
        elif vertical == DIRECTION_DOWN and horizontal == DIRECTION_CENTER:
            return DIRECTION_DOWN
        elif horizontal == DIRECTION_LEFT and vertical == DIRECTION_CENTER:
            return DIRECTION_LEFT
        elif horizontal == DIRECTION_RIGHT and vertical == DIRECTION_CENTER:
            return DIRECTION_RIGHT
        elif horizontal == DIRECTION_LEFT and vertical == DIRECTION_UP:
            return "up_left"
        elif horizontal == DIRECTION_RIGHT and vertical == DIRECTION_UP:
            return "up_right"
        elif horizontal == DIRECTION_LEFT and vertical == DIRECTION_DOWN:
            return "down_left"
        elif horizontal == DIRECTION_RIGHT and vertical == DIRECTION_DOWN:
            return "down_right"
        else:
            return DIRECTION_CENTER

    def _process_frame(self, frame: np.ndarray) -> GazeBoostState:
        """
        处理图像帧，检测视线方向

        Args:
            frame: 输入图像帧

        Returns:
            GazeBoostState: 视线方向状态
        """
        # 创建新状态
        state = GazeBoostState(frame=frame, timestamp=time.time())

        # 转换为RGB进行MediaPipe处理
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        # 检查是否检测到人脸
        if not results.multi_face_landmarks:
            return state

        # 只处理第一张人脸
        face_landmarks = results.multi_face_landmarks[0]

        # 提取特征
        features, avg_h_ratio, avg_v_ratio, left_eye_pos, right_eye_pos = self._extract_features(
            face_landmarks, frame.shape)

        # 预测视线方向
        horizontal_dir, vertical_dir, h_confidence, v_confidence = self._predict_gaze(
            features)

        # 确定组合方向
        combined_dir = self._get_combined_direction(
            horizontal_dir, vertical_dir)

        # 更新状态
        gaze_direction = state.detections["gaze_direction"]
        gaze_direction["horizontal"]["direction"] = horizontal_dir
        gaze_direction["horizontal"]["confidence"] = float(h_confidence)
        gaze_direction["vertical"]["direction"] = vertical_dir
        gaze_direction["vertical"]["confidence"] = float(v_confidence)
        gaze_direction["combined_direction"] = combined_dir
        gaze_direction["iris_position"]["horizontal_ratio"] = float(
            avg_h_ratio)
        gaze_direction["iris_position"]["vertical_ratio"] = float(avg_v_ratio)
        gaze_direction["iris_position"]["left_eye"] = (
            float(left_eye_pos[0]), float(left_eye_pos[1]))
        gaze_direction["iris_position"]["right_eye"] = (
            float(right_eye_pos[0]), float(right_eye_pos[1]))
        gaze_direction["face_detected"] = True

        # 如果启用调试模式，在帧上绘制信息
        if self.debug and frame is not None:
            self._draw_debug_info(frame, state)

        return state

    def _draw_debug_info(self, frame: np.ndarray, state: GazeBoostState):
        """
        在图像帧上绘制调试信息

        Args:
            frame: 图像帧
            state: 视线方向状态
        """
        gaze_info = state.detections["gaze_direction"]

        # 如果没有检测到人脸，不绘制
        if not gaze_info["face_detected"]:
            return

        # 绘制方向信息
        h_dir = gaze_info["horizontal"]["direction"]
        v_dir = gaze_info["vertical"]["direction"]
        h_conf = gaze_info["horizontal"]["confidence"]
        v_conf = gaze_info["vertical"]["confidence"]
        combined_dir = gaze_info["combined_direction"]

        # 绘制数据
        cv2.putText(frame, f"水平: {h_dir} ({h_conf:.2f})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"垂直: {v_dir} ({v_conf:.2f})", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"组合: {combined_dir}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 绘制原始虹膜位置
        h_ratio = gaze_info["iris_position"]["horizontal_ratio"]
        v_ratio = gaze_info["iris_position"]["vertical_ratio"]
        cv2.putText(frame, f"水平比例: {h_ratio:.2f}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        cv2.putText(frame, f"垂直比例: {v_ratio:.2f}", (10, 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

        # 绘制方向指示器
        self._draw_gaze_direction_indicator(
            frame, h_dir, v_dir, h_conf, v_conf)

    def _draw_gaze_direction_indicator(self, frame, horizontal_dir, vertical_dir, h_conf, v_conf):
        """
        在屏幕上绘制视线方向指示器

        Args:
            frame: 图像帧
            horizontal_dir: 水平方向
            vertical_dir: 垂直方向
            h_conf: 水平置信度
            v_conf: 垂直置信度
        """
        h, w = frame.shape[:2]

        # 绘制方向指示器背景
        indicator_size = 150
        margin = 20
        indicator_x = w - indicator_size - margin
        indicator_y = margin

        # 画一个黑色背景
        cv2.rectangle(frame,
                      (indicator_x, indicator_y),
                      (indicator_x + indicator_size, indicator_y + indicator_size),
                      (0, 0, 0), -1)

        # 绘制边框
        cv2.rectangle(frame,
                      (indicator_x, indicator_y),
                      (indicator_x + indicator_size, indicator_y + indicator_size),
                      (200, 200, 200), 2)

        # 计算中心和偏移
        center_x = indicator_x + indicator_size // 2
        center_y = indicator_y + indicator_size // 2

        # 绘制方向指示
        offset = indicator_size // 3

        # 绘制中心点
        cv2.circle(frame, (center_x, center_y), 5, (100, 100, 100), -1)

        # 根据水平方向设置X偏移
        if horizontal_dir == "left":
            dir_x = center_x - offset
        elif horizontal_dir == "right":
            dir_x = center_x + offset
        else:  # center
            dir_x = center_x

        # 根据垂直方向设置Y偏移
        if vertical_dir == "up":
            dir_y = center_y - offset
        elif vertical_dir == "down":
            dir_y = center_y + offset
        else:  # center
            dir_y = center_y

        # 绘制当前视线方向点
        cv2.circle(frame, (dir_x, dir_y), 10,
                   self.DIRECTION_COLORS.get(horizontal_dir, (0, 255, 0)), -1)

        # 添加方向文本
        cv2.putText(frame,
                    f"H: {horizontal_dir} ({h_conf:.2f})",
                    (indicator_x + 5, indicator_y + indicator_size + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.putText(frame,
                    f"V: {vertical_dir} ({v_conf:.2f})",
                    (indicator_x + 5, indicator_y + indicator_size + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def update(self) -> GazeBoostState:
        """
        获取最新的视线方向状态

        Returns:
            GazeBoostState: 当前视线方向状态
        """
        try:
            ret, frame = self.camera_manager.read_frame()
            if not ret:
                if self.camera_manager.config.is_file_source and self.loop_video:
                    self.camera_manager.capture_state.capture.set(
                        cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.camera_manager.read_frame()
                    if not ret:
                        logger.error("无法获取视频帧")
                        return GazeBoostState()
                else:
                    logger.error("无法获取视频帧")
                    return GazeBoostState()

            return self._process_frame(frame)
        except (IOError, RuntimeError, ValueError) as e:
            logger.error("处理帧时出错: %s", str(e), exc_info=True)
            return GazeBoostState()

    def start(self) -> int:
        """
        开始视线方向跟踪

        Returns:
            int: 错误码，0表示成功，其他值表示失败
        """
        result = super().start()
        if result != SUCCESS:
            logger.error("无法启动增强视线方向跟踪器: %d", result)
            return result

        logger.info("增强视线方向跟踪器已开始运行")
        return SUCCESS

    def shutdown(self) -> int:
        """
        关闭视线方向跟踪资源

        Returns:
            int: 错误码，0表示成功，其他值表示失败
        """
        try:
            if self.face_mesh:
                self.face_mesh.close()
                self.face_mesh = None
                logger.info("MediaPipe资源已释放")

            # 释放模型资源
            self.horizontal_model = None
            self.vertical_model = None
            logger.info("模型资源已释放")

            result = super().shutdown()
            if result == SUCCESS:
                logger.info("增强视线方向跟踪器已关闭")
            else:
                logger.error("关闭增强视线方向跟踪器时出错: %d", result)

            return result

        except (IOError, RuntimeError) as e:
            logger.error("关闭增强视线方向跟踪器失败: %s", str(e), exc_info=True)
            return RUNTIME_ERROR

    def get_key_info(self) -> str:
        """
        获取模态的关键信息

        Returns:
            str: 模态的关键信息
        """
        state = self.update()
        key_info = None

        if state.frame is not None:
            gaze_info = state.detections["gaze_direction"]

            if gaze_info["face_detected"]:
                horizontal_dir = gaze_info["horizontal"]["direction"]
                vertical_dir = gaze_info["vertical"]["direction"]
                combined_dir = gaze_info["combined_direction"]

                if horizontal_dir == DIRECTION_CENTER and vertical_dir == DIRECTION_CENTER:
                    key_info = "中间"
                else:
                    key_info = "非中间"

        return key_info
