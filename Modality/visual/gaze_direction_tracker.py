"""
视线方向跟踪模块，用于检测和跟踪用户的视线方向。
该模块使用MediaPipe人脸网格检测眼睛和虹膜位置，并计算视线方向。
"""
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import cv2
import mediapipe as mp
import numpy as np

from modality.core.error_codes import (MEDIAPIPE_INITIALIZATION_FAILED,
                                       RUNTIME_ERROR, SUCCESS)
from modality.visual.base_visual import BaseVisualModality, VisualState

logging.basicConfig(
    level=logging.DEBUG if os.environ.get(
        'MODALITY_DEBUG', '0') == '1' else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='./database/log/gaze_direction_tracker.log',
    filemode='w',
    encoding='utf-8'
)
logger = logging.getLogger('GazeDirectionTracker')

DEBUG = os.environ.get('MODALITY_DEBUG', '0') == '1'

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 视线方向常量
DIRECTION_CENTER = "center"
DIRECTION_LEFT = "left"
DIRECTION_RIGHT = "right"
DIRECTION_UP = "up"
DIRECTION_DOWN = "down"

# 视线检测参数
LEFT_EYE_INDICES = [362, 382, 381, 380, 374, 373,
                    390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154,
                     155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_IRIS_INDICES = [468, 469, 470, 471, 472]
RIGHT_IRIS_INDICES = [473, 474, 475, 476, 477]

HORIZONTAL_RATIO_THRESHOLD = 0.45
VERTICAL_RATIO_THRESHOLD = 0.45
CENTER_THRESHOLD = 0.20

VIDEO_FPS = 30


@dataclass
class EyeData:
    """眼睛关键点数据"""
    eye_landmarks: List[Tuple[int, int]] = field(default_factory=list)
    iris_landmarks: List[Tuple[int, int]] = field(default_factory=list)
    iris_position: Tuple[float, float] = (0.0, 0.0)  # (水平比例, 垂直比例)


@dataclass
class GazeData:
    """视线方向数据"""
    direction: str = DIRECTION_CENTER
    confidence: float = 0.0
    horizontal_ratio: float = 0.0
    vertical_ratio: float = 0.0


@dataclass
class GazeDirectionState(VisualState):
    """
    视线方向状态类，扩展自VisualState，存储检测到的视线方向信息
    """

    def __init__(self, frame=None, timestamp=None):
        super().__init__(frame, timestamp)
        self.detections = {
            "gaze_direction": {
                "direction": DIRECTION_CENTER,   # 视线方向
                "confidence": 0.0,               # 置信度
                "horizontal_ratio": 0.0,         # 水平比例(-1到1，负为左，正为右)
                "vertical_ratio": 0.0,           # 垂直比例(-1到1，负为上，正为下)
                "left_eye": {
                    "iris_position": (0.0, 0.0),  # 左眼虹膜位置比例
                    "eye_landmarks": []           # 左眼关键点
                },
                "right_eye": {
                    "iris_position": (0.0, 0.0),  # 右眼虹膜位置比例
                    "eye_landmarks": []           # 右眼关键点
                },
                "face_detected": False            # 是否检测到人脸
            }
        }

    def to_dict(self) -> Dict[str, Any]:
        """将状态转换为字典格式"""
        result = super().to_dict()
        result.update({"gaze tracection": self.detections["gaze_direction"]})
        return result

    def is_looking_center(self) -> bool:
        """检查用户是否注视中央"""
        return self.detections["gaze_direction"]["direction"] == DIRECTION_CENTER

    def get_direction(self) -> str:
        """获取当前视线方向"""
        return self.detections["gaze_direction"]["direction"]


class GazeDirectionTracker(BaseVisualModality):
    """
    视线方向跟踪器，检测用户目光注视的大致方向（上、下、左、右、中）
    """

    def __init__(self,
                 name: str = "gaze_direction_tracker",
                 source: int = 0,
                 width: int = 640,
                 height: int = 480,
                 **kwargs):
        """
        初始化视线方向跟踪器

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

        self._processing_thread = None
        self._stop_event = threading.Event()

        self._latest_state = GazeDirectionState()
        self._state_lock = threading.Lock()

        self.face_mesh = None

    def initialize(self) -> int:
        """
        初始化摄像头和MediaPipe人脸网格

        Returns:
            int: 错误码，0表示成功，其他值表示失败
        """
        result = super().initialize()
        if result != SUCCESS:
            logger.error("基础视觉模态初始化失败")
            return result

        try:
            self.face_mesh = mp_face_mesh.FaceMesh(
                max_num_faces=self.mp_face_options['max_num_faces'],
                refine_landmarks=self.mp_face_options['refine_landmarks'],
                min_detection_confidence=self.mp_face_options['min_detection_confidence'],
                min_tracking_confidence=self.mp_face_options['min_tracking_confidence'],
            )
            logger.info("MediaPipe人脸网格初始化成功")

            return SUCCESS
        except (ValueError, RuntimeError, ImportError) as e:
            logger.error("初始化失败: %s", str(e), exc_info=True)
            return MEDIAPIPE_INITIALIZATION_FAILED

    def _processing_loop(self):
        """处理线程的主循环，负责连续处理视频帧"""
        logger.info("处理线程已启动")

        if self.camera_manager is None:
            logger.error("视频源未初始化")
            return

        frame_interval = 1.0 / VIDEO_FPS

        while not self._stop_event.is_set():
            start_time = time.time()

            # ret, frame = self.capture.read()
            ret, frame = self.camera_manager.read_frame()
            if not ret:
                if self.camera_manager.config.is_file_source and self.loop_video:
                    # self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.camera_manager.capture_state.capture.set(
                        cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                logger.error("无法获取视频帧")
                break

            state = self._process_frame(frame)
            with self._state_lock:
                self._latest_state = state

            elapsed_time = time.time() - start_time
            sleep_time = max(0, frame_interval - elapsed_time)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def update(self) -> GazeDirectionState:
        """
        获取最新的视线方向状态

        Returns:
            GazeDirectionState: 当前视线方向状态
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
                        return GazeDirectionState()
                else:
                    logger.error("无法获取视频帧")
                    return GazeDirectionState()

            return self._process_frame(frame)
        except (IOError, RuntimeError, ValueError) as e:
            logger.error("处理帧时出错: %s", str(e), exc_info=True)
            return GazeDirectionState()

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

        # 使用生成器表达式优化求和操作
        iris_center_x = sum(p[0] for p in iris_landmarks) / len(iris_landmarks)
        iris_center_y = sum(p[1] for p in iris_landmarks) / len(iris_landmarks)

        horizontal_ratio = 2 * (iris_center_x - eye_left) / eye_width - 1
        vertical_ratio = 2 * (iris_center_y - eye_top) / eye_height - 1

        return horizontal_ratio, vertical_ratio

    def _determine_gaze_direction(self, left_h_ratio, left_v_ratio, right_h_ratio, right_v_ratio):
        """
        根据两只眼睛的虹膜位置确定视线方向

        Args:
            left_h_ratio: 左眼水平比例
            left_v_ratio: 左眼垂直比例
            right_h_ratio: 右眼水平比例
            right_v_ratio: 右眼垂直比例

        Returns:
            Tuple[str, float, float, float]: 方向，置信度，平均水平比例，平均垂直比例
        """
        avg_h_ratio = (left_h_ratio + right_h_ratio) / 2
        avg_v_ratio = (left_v_ratio + right_v_ratio) / 2

        h_threshold = HORIZONTAL_RATIO_THRESHOLD
        v_threshold = VERTICAL_RATIO_THRESHOLD
        center_threshold = CENTER_THRESHOLD

        confidence = max(abs(avg_h_ratio), abs(avg_v_ratio))

        if abs(avg_h_ratio) <= center_threshold and abs(avg_v_ratio) <= center_threshold:
            return DIRECTION_CENTER, confidence, avg_h_ratio, avg_v_ratio

        if abs(avg_h_ratio) >= h_threshold:
            if avg_h_ratio > 0:
                return DIRECTION_RIGHT, confidence, avg_h_ratio, avg_v_ratio
            return DIRECTION_LEFT, confidence, avg_h_ratio, avg_v_ratio

        if abs(avg_v_ratio) >= v_threshold:
            if avg_v_ratio > 0:
                return DIRECTION_DOWN, confidence, avg_h_ratio, avg_v_ratio
            return DIRECTION_UP, confidence, avg_h_ratio, avg_v_ratio

        return DIRECTION_CENTER, confidence, avg_h_ratio, avg_v_ratio

    def _extract_landmarks(self, face_landmarks, frame_width, frame_height):
        """
        从面部特征点提取眼睛和虹膜的关键点

        Args:
            face_landmarks: 面部特征点
            frame_width: 帧宽度
            frame_height: 帧高度

        Returns:
            Tuple: 包含左眼、右眼、左虹膜和右虹膜关键点的元组
        """
        left_eye_landmarks = []
        right_eye_landmarks = []
        left_iris_landmarks = []
        right_iris_landmarks = []

        for idx, landmark in enumerate(face_landmarks.landmark):
            x, y = landmark.x, landmark.y
            pixel_x, pixel_y = int(x * frame_width), int(y * frame_height)

            if idx in LEFT_EYE_INDICES:
                left_eye_landmarks.append((pixel_x, pixel_y))
            if idx in RIGHT_EYE_INDICES:
                right_eye_landmarks.append((pixel_x, pixel_y))
            if idx in LEFT_IRIS_INDICES:
                left_iris_landmarks.append((pixel_x, pixel_y))
            if idx in RIGHT_IRIS_INDICES:
                right_iris_landmarks.append((pixel_x, pixel_y))

        return left_eye_landmarks, right_eye_landmarks, left_iris_landmarks, right_iris_landmarks

    def _process_frame(self, frame: np.ndarray) -> GazeDirectionState:
        """
        处理图像帧，检测视线方向

        Args:
            frame: 输入图像帧

        Returns:
            GazeDirectionState: 视线方向状态
        """
        state = GazeDirectionState(frame=frame, timestamp=time.time())

        # 处理图像
        results = self._process_image(frame)
        if not results or not results.multi_face_landmarks:
            return state

        # 提取面部关键点
        face_landmarks = results.multi_face_landmarks[0]
        h, w, _ = frame.shape

        # 提取并验证眼睛关键点
        left_eye_data, right_eye_data = self._extract_eye_data(
            face_landmarks, w, h)
        if not left_eye_data or not right_eye_data:
            return state

        # 计算视线方向
        gaze_data = self._calculate_gaze_direction(
            left_eye_data, right_eye_data)

        # 更新状态
        self._update_state(state, left_eye_data, right_eye_data, gaze_data)

        # 如果启用调试模式，在帧上绘制信息
        if self.debug:
            self._draw_debug_info(frame, state)

        return state

    def _process_image(self, frame: np.ndarray):
        """处理图像并返回MediaPipe结果"""
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.face_mesh.process(image_rgb)

    def _extract_eye_data(self, face_landmarks, frame_width, frame_height):
        """提取并验证眼睛关键点数据"""
        left_eye, right_eye, left_iris, right_iris = self._extract_landmarks(
            face_landmarks, frame_width, frame_height)

        # 检查是否提取了所有需要的关键点
        if (len(left_eye) != len(LEFT_EYE_INDICES) or
            len(right_eye) != len(RIGHT_EYE_INDICES) or
            len(left_iris) != len(LEFT_IRIS_INDICES) or
                len(right_iris) != len(RIGHT_IRIS_INDICES)):
            return None, None

        # 创建眼睛数据对象
        left_eye_data = EyeData(eye_landmarks=left_eye,
                                iris_landmarks=left_iris)
        right_eye_data = EyeData(
            eye_landmarks=right_eye, iris_landmarks=right_iris)

        # 计算虹膜位置
        left_h_ratio, left_v_ratio = self._calculate_iris_position(
            left_eye, left_iris)
        right_h_ratio, right_v_ratio = self._calculate_iris_position(
            right_eye, right_iris)

        left_eye_data.iris_position = (left_h_ratio, left_v_ratio)
        right_eye_data.iris_position = (right_h_ratio, right_v_ratio)

        return left_eye_data, right_eye_data

    def _calculate_gaze_direction(self, left_eye_data: EyeData, right_eye_data: EyeData):
        """根据眼睛数据计算视线方向"""
        left_h_ratio, left_v_ratio = left_eye_data.iris_position
        right_h_ratio, right_v_ratio = right_eye_data.iris_position

        direction, confidence, avg_h_ratio, avg_v_ratio = self._determine_gaze_direction(
            left_h_ratio, left_v_ratio, right_h_ratio, right_v_ratio
        )

        return GazeData(
            direction=direction,
            confidence=confidence,
            horizontal_ratio=avg_h_ratio,
            vertical_ratio=avg_v_ratio
        )

    def _update_state(self, state: GazeDirectionState,
                      left_eye_data: EyeData, right_eye_data: EyeData,
                      gaze_data: GazeData):
        """更新状态对象"""
        state_gaze = state.detections["gaze_direction"]

        # 更新方向数据
        state_gaze["direction"] = gaze_data.direction
        state_gaze["confidence"] = float(gaze_data.confidence)
        state_gaze["horizontal_ratio"] = float(gaze_data.horizontal_ratio)
        state_gaze["vertical_ratio"] = float(gaze_data.vertical_ratio)

        # 更新眼睛数据
        state_gaze["left_eye"]["iris_position"] = (
            float(left_eye_data.iris_position[0]), float(left_eye_data.iris_position[1]))
        state_gaze["right_eye"]["iris_position"] = (
            float(right_eye_data.iris_position[0]), float(right_eye_data.iris_position[1]))

        state_gaze["left_eye"]["eye_landmarks"] = [
            (float(x), float(y)) for x, y in left_eye_data.eye_landmarks]
        state_gaze["right_eye"]["eye_landmarks"] = [
            (float(x), float(y)) for x, y in right_eye_data.eye_landmarks]

        state_gaze["face_detected"] = True

    def _draw_debug_info(self, frame: np.ndarray, state: GazeDirectionState):
        """
        绘制调试信息到视频帧上

        Args:
            frame: 视频帧
            state: 视线方向状态
        """
        gaze_info = state.detections["gaze_direction"]
        if not gaze_info["face_detected"]:
            return

        # 提取需要的数据
        left_eye_landmarks = gaze_info["left_eye"]["eye_landmarks"]
        right_eye_landmarks = gaze_info["right_eye"]["eye_landmarks"]

        # 绘制眼睛轮廓点
        for point in left_eye_landmarks:
            cv2.circle(frame, (int(point[0]), int(
                point[1])), 2, (0, 255, 0), -1)
        for point in right_eye_landmarks:
            cv2.circle(frame, (int(point[0]), int(
                point[1])), 2, (0, 255, 0), -1)

        # 显示方向和比例信息
        direction = gaze_info["direction"]
        confidence = gaze_info["confidence"]
        h_ratio = gaze_info["horizontal_ratio"]
        v_ratio = gaze_info["vertical_ratio"]

        direction_text = f"方向: {direction}, 置信度: {confidence:.2f}"
        cv2.putText(frame, direction_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        ratio_text = f"水平: {h_ratio:.2f}, 垂直: {v_ratio:.2f}"
        cv2.putText(frame, ratio_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    def start(self) -> int:
        """
        开始视线方向跟踪

        Returns:
            int: 错误码，0表示成功，其他值表示失败
        """
        result = super().start()
        if result != SUCCESS:
            logger.error("无法启动视线方向跟踪器: %d", result)
            return result

        logger.info("视线方向跟踪器已开始运行")
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

            result = super().shutdown()
            if result == SUCCESS:
                logger.info("视线方向跟踪器已关闭")
            else:
                logger.error("关闭视线方向跟踪器时出错: %d", result)

            return result

        except (IOError, RuntimeError) as e:
            logger.error("关闭视线方向跟踪器失败: %s", str(e), exc_info=True)
            return RUNTIME_ERROR

    def get_key_info(self) -> str:
        """
        获取模态的关键信息

        Returns:
            str: 模态的关键信息
        """
        key_info = None
        state = self.update()
        frame = state.frame
        if frame is not None:
            gaze_info = state.detections["gaze_direction"]

            if gaze_info["face_detected"]:
                direction = gaze_info["direction"]
                confidence = gaze_info["confidence"]
                # print(f"方向: {direction_names.get(direction, '未知')}, 置信度: {confidence:.2f}")

                if confidence < 1.0 or direction == DIRECTION_CENTER:
                    key_info = "中间"
                else:
                    key_info = "非中间"
        # print(f"视线key_info: {key_info}")
        return key_info
