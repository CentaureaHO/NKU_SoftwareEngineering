import cv2
import numpy as np
from typing import Dict, Any, List, Tuple
import time
import logging
import os
import threading
from collections import deque
import sys
import math

logging.basicConfig(
    level=logging.DEBUG if os.environ.get('MODALITY_DEBUG', '0') == '1' else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='gaze_direction_tracker.log',
    filemode='w'
)
logger = logging.getLogger('GazeDirectionTracker')

DEBUG = os.environ.get('MODALITY_DEBUG', '0') == '1'

import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Modality.visual.base_visual import BaseVisualModality, VisualState
from Modality.core.error_codes import (
    SUCCESS, MEDIAPIPE_INITIALIZATION_FAILED, RUNTIME_ERROR
)

DIRECTION_CENTER = "center"
DIRECTION_LEFT = "left"
DIRECTION_RIGHT = "right"
DIRECTION_UP = "up"
DIRECTION_DOWN = "down"
DIRECTION_UP_LEFT = "up_left"
DIRECTION_UP_RIGHT = "up_right"
DIRECTION_DOWN_LEFT = "down_left"
DIRECTION_DOWN_RIGHT = "down_right"

class GazeParams:
    LEFT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    RIGHT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    LEFT_IRIS_INDICES = [468, 469, 470, 471, 472]
    RIGHT_IRIS_INDICES = [473, 474, 475, 476, 477]
    
    # 平衡垂直方向敏感度参数
    HORIZONTAL_RATIO_THRESHOLD = 0.45
    VERTICAL_RATIO_THRESHOLD = 0.38      # 提高垂直阈值，降低敏感度
    CENTER_THRESHOLD = 0.18              # 扩大中心区域
    VERTICAL_BOOST = 1.0                 # 移除垂直方向信号增强
    
    # 创建更宽松的向上判断条件
    UP_THRESHOLD_STRICTER = -0.45        # 更严格的向上阈值
    DOWN_THRESHOLD_STRICTER = 0.38       # 下方阈值
    
    # 瞳孔检测参数
    PUPIL_DETECTION_THRESHOLD = 0.01     # 瞳孔检测面积比例阈值
    EYE_SPHERE_RADIUS = 15.0             # 眼球球面模型半径
    DEPTH_SCALE_FACTOR = 0.003           # 深度缩放因子
    
    # 平滑参数
    SMOOTHING_WINDOW_SIZE = 10           # 增加平滑窗口大小进一步稳定
    POSITION_ALPHA = 0.25                # 降低位置平滑系数，更平滑
    DIRECTION_ALPHA = 0.3                # 降低方向平滑系数
    
    # 视线一致性检测阈值
    EYE_CONSISTENCY_THRESHOLD = 0.25     # 左右眼一致性检查阈值
    VERTICAL_CONSISTENCY_WEIGHT = 0.7    # 增加垂直方向一致性权重
    HORIZONTAL_CONSISTENCY_WEIGHT = 0.3  # 降低水平方向一致性权重
    
    # 组合视线计算权重
    LEFT_EYE_WEIGHT = 0.5                # 左眼权重
    RIGHT_EYE_WEIGHT = 0.5               # 右眼权重
    
    # 增加垂直方向偏移修正
    VERTICAL_OFFSET = 0.05               # 垂直方向向下偏移量，修正向上偏差
    
    VIDEO_FPS = 30

class GazeDirectionState(VisualState):
    """视线方向状态类，扩展自VisualState"""
    
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
        result = super().to_dict()
        return result

class GazeDirectionTracker(BaseVisualModality):
    """
    视线方向跟踪器，检测用户目光注视的大致方向（上、下、左、右、中）
    """
    
    def __init__(self, name: str = "gaze_direction_tracker", source: int = 0, 
                 width: int = 640, height: int = 480,
                 min_detection_confidence: float = 0.5, 
                 min_tracking_confidence: float = 0.5,
                 debug: bool = DEBUG):
        """
        初始化视线方向跟踪器
        
        Args:
            name: 模态名称
            source: 视频源，可以是摄像头ID或视频文件路径
            width: 图像宽度
            height: 图像高度
            min_detection_confidence: 检测置信度阈值
            min_tracking_confidence: 跟踪置信度阈值
            debug: 是否启用调试模式
        """
        super().__init__(name, source, width, height)

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
        self.debug = debug
        
        # 增强平滑处理的数据结构
        self.left_h_ratio_history = deque(maxlen=GazeParams.SMOOTHING_WINDOW_SIZE)
        self.left_v_ratio_history = deque(maxlen=GazeParams.SMOOTHING_WINDOW_SIZE)
        self.right_h_ratio_history = deque(maxlen=GazeParams.SMOOTHING_WINDOW_SIZE)
        self.right_v_ratio_history = deque(maxlen=GazeParams.SMOOTHING_WINDOW_SIZE)
        self.left_pupil_center_history = deque(maxlen=GazeParams.SMOOTHING_WINDOW_SIZE)
        self.right_pupil_center_history = deque(maxlen=GazeParams.SMOOTHING_WINDOW_SIZE)
        self.left_gaze_direction_history = deque(maxlen=GazeParams.SMOOTHING_WINDOW_SIZE)
        self.right_gaze_direction_history = deque(maxlen=GazeParams.SMOOTHING_WINDOW_SIZE)
        
        # 动态阈值计算相关变量
        self.h_ratio_variance = 0.1  # 初始水平方向比例方差
        self.v_ratio_variance = 0.1  # 初始垂直方向比例方差
    
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
        except Exception as e:
            logger.error(f"初始化失败: {str(e)}")
            return MEDIAPIPE_INITIALIZATION_FAILED
    
    def _processing_loop(self):
        """处理线程的主循环，负责连续处理视频帧"""
        logger.info("处理线程已启动")
        
        if self.capture is None:
            logger.error("视频源未初始化")
            return
        
        frame_interval = 1.0 / GazeParams.VIDEO_FPS
        
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
    
    def update(self) -> GazeDirectionState:
        """
        获取最新的视线方向状态
        
        Returns:
            GazeDirectionState: 当前视线方向状态
        """
        if not self._is_running or self.capture is None:
            logger.warning("视频源未运行")
            return GazeDirectionState()
        
        try:
            ret, frame = self.capture.read()
            if not ret:
                if self.is_file_source and self.loop_video:
                    self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.capture.read()
                    if not ret:
                        logger.error("无法获取视频帧")
                        return GazeDirectionState()
                else:
                    logger.error("无法获取视频帧")
                    return GazeDirectionState()
                    
            return self._process_frame(frame)
        except Exception as e:
            logger.error(f"处理帧时出错: {str(e)}")
            return GazeDirectionState()
    
    def _smooth_value(self, value_history, new_value, alpha=None):
        """
        使用指数移动平均平滑数值
        
        Args:
            value_history: 历史值队列
            new_value: 新值
            alpha: 可选的自定义平滑系数
            
        Returns:
            float: 平滑后的值
        """
        if not value_history:
            return new_value
        
        if alpha is None:
            alpha = GazeParams.POSITION_ALPHA
            
        # 使用指数加权移动平均
        if isinstance(new_value, tuple):
            prev_x, prev_y = value_history[-1]
            curr_x, curr_y = new_value
            return (alpha * curr_x + (1 - alpha) * prev_x,
                   alpha * curr_y + (1 - alpha) * prev_y)
        else:
            return alpha * new_value + (1 - alpha) * value_history[-1]
    
    def _enhanced_iris_detection(self, eye_region, landmarks):
        """
        改进的虹膜检测方法，结合图像处理技术提高精度
        
        Args:
            eye_region: 眼部区域图像
            landmarks: 眼部关键点
            
        Returns:
            tuple: 虹膜中心坐标 (x, y) 或 None
        """
        try:
            if eye_region.size == 0:
                return None
                
            # 转为灰度图
            gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
            
            # 计算瞳孔的大致中心位置(作为初始估计)
            eye_x_coords = [p[0] for p in landmarks]
            eye_y_coords = [p[1] for p in landmarks]
            eye_left = min(eye_x_coords)
            eye_top = min(eye_y_coords)
            rel_x_coords = [x - eye_left for x in eye_x_coords]
            rel_y_coords = [y - eye_top for y in eye_y_coords]
            
            # 自适应阈值处理
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # 使用多种阈值方法尝试找到最佳效果
            _, thresh1 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            thresh2 = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY_INV, 11, 2)
            
            # 组合两种阈值结果
            thresh = cv2.bitwise_and(thresh1, thresh2)
            
            # 形态学操作改善分割结果
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
            
            # 找轮廓
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 面积过滤，找出可能是虹膜的轮廓
            min_area = eye_region.shape[0] * eye_region.shape[1] * GazeParams.PUPIL_DETECTION_THRESHOLD
            valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
            
            if valid_contours:
                # 找到最黑暗的区域作为瞳孔
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
        """
        计算眼部中心点
        
        Args:
            eye_landmarks: 眼部关键点列表
            
        Returns:
            tuple: 眼部中心(x, y)
        """
        x_coords = [p[0] for p in eye_landmarks]
        y_coords = [p[1] for p in eye_landmarks]
        return (sum(x_coords) / len(x_coords), sum(y_coords) / len(y_coords))
    
    def _calculate_3d_gaze_direction(self, pupil_center, eye_center, image_height):
        """
        计算3D视线方向向量
        
        Args:
            pupil_center: 瞳孔中心坐标
            eye_center: 眼部中心坐标
            image_height: 图像高度(用于深度比例计算)
            
        Returns:
            tuple: 3D方向向量(dx, dy, dz)，归一化
        """
        if not pupil_center or not eye_center:
            return None
            
        # 计算从眼中心到瞳孔中心的2D向量
        ox, oy = eye_center
        px, py = pupil_center
        dx_2d = px - ox
        dy_2d = py - oy
        
        # 2D距离
        distance_2d = math.sqrt(dx_2d ** 2 + dy_2d ** 2)
        if distance_2d < 1e-6:
            return (0, 0, 1)  # 正视前方
            
        # 深度缩放
        depth_scale = image_height * GazeParams.DEPTH_SCALE_FACTOR
        
        # 使用眼球球面模型计算z分量
        r_squared = GazeParams.EYE_SPHERE_RADIUS ** 2
        d_squared = (distance_2d / depth_scale) ** 2
        
        # 确保值在有效范围内
        value = r_squared - d_squared
        dz = math.sqrt(max(0.1, value)) if value >= 0 else 0.1
        
        # 归一化向量
        norm = math.sqrt(dx_2d ** 2 + dy_2d ** 2 + (dz * depth_scale) ** 2)
        
        return (dx_2d / norm, dy_2d / norm, (dz * depth_scale) / norm)
    
    def _extract_eye_region(self, eye_landmarks, frame):
        """
        提取眼部区域图像
        
        Args:
            eye_landmarks: 眼部关键点
            frame: 图像帧
            
        Returns:
            tuple: (眼部区域图像, 左上角偏移)
        """
        eye_x_coords = [p[0] for p in eye_landmarks]
        eye_y_coords = [p[1] for p in eye_landmarks]
        
        x1 = max(0, int(min(eye_x_coords)) - 5)
        y1 = max(0, int(min(eye_y_coords)) - 5)
        x2 = min(frame.shape[1], int(max(eye_x_coords)) + 5)
        y2 = min(frame.shape[0], int(max(eye_y_coords)) + 5)
        
        # 确保区域有效
        if x2 <= x1 or y2 <= y1:
            return None, (x1, y1)
            
        eye_region = frame[y1:y2, x1:x2]
        return eye_region, (x1, y1)
    
    def _calculate_iris_position(self, eye_landmarks, iris_landmarks, frame=None):
        """
        计算虹膜相对于眼睛的位置，增强版
        
        Args:
            eye_landmarks: 眼睛轮廓关键点
            iris_landmarks: 虹膜关键点
            frame: 图像帧
            
        Returns:
            Tuple[float, float, tuple, tuple]: 水平比例, 垂直比例, 3D视线向量, 瞳孔中心
        """
        eye_x_coords = [p[0] for p in eye_landmarks]
        eye_y_coords = [p[1] for p in eye_landmarks]
        
        eye_left = min(eye_x_coords)
        eye_right = max(eye_x_coords)
        eye_top = min(eye_y_coords)
        eye_bottom = max(eye_y_coords)
        
        eye_width = max(eye_right - eye_left, 1e-5)
        eye_height = max(eye_bottom - eye_top, 1e-5)
        
        # 计算眼睛中心点
        eye_center = self._calculate_eye_center(eye_landmarks)
        
        # 初始虹膜中心（使用MediaPipe检测的坐标）
        iris_center_x = sum([p[0] for p in iris_landmarks]) / len(iris_landmarks)
        iris_center_y = sum([p[1] for p in iris_landmarks]) / len(iris_landmarks)
        iris_center = (iris_center_x, iris_center_y)
        
        # 使用图像处理增强虹膜检测
        if frame is not None:
            eye_region, offset = self._extract_eye_region(eye_landmarks, frame)
            if eye_region is not None and eye_region.size > 0:
                enhanced_iris = self._enhanced_iris_detection(eye_region, 
                    [(p[0] - offset[0], p[1] - offset[1]) for p in eye_landmarks])
                
                if enhanced_iris:
                    iris_center = (enhanced_iris[0] + offset[0], enhanced_iris[1] + offset[1])
                    iris_center_x, iris_center_y = iris_center
        
        # 计算比例并归一化到 [-1, 1] 范围
        horizontal_ratio = 2 * (iris_center_x - eye_left) / eye_width - 1
        vertical_ratio = 2 * (iris_center_y - eye_top) / eye_height - 1
        
        # 移除垂直方向的增强，并添加偏移修正（解决偏向上方的问题）
        vertical_ratio = vertical_ratio + GazeParams.VERTICAL_OFFSET
        
        # 限制在 [-1, 1] 范围内
        vertical_ratio = max(-1.0, min(1.0, vertical_ratio))
        horizontal_ratio = max(-1.0, min(1.0, horizontal_ratio))
        
        # 计算3D视线方向
        gaze_3d = self._calculate_3d_gaze_direction(
            iris_center, eye_center, frame.shape[0] if frame is not None else 480)
            
        return horizontal_ratio, vertical_ratio, gaze_3d, iris_center
    
    def _check_eye_consistency(self, left_h_ratio, left_v_ratio, right_h_ratio, right_v_ratio):
        """
        检查左右眼视线方向是否一致，改进版
        
        Args:
            left_h_ratio: 左眼水平比例
            left_v_ratio: 左眼垂直比例
            right_h_ratio: 右眼水平比例
            right_v_ratio: 右眼垂直比例
            
        Returns:
            tuple: (是否一致, 一致性分数)
        """
        # 分别计算水平和垂直方向的一致性
        h_diff = abs(left_h_ratio - right_h_ratio)
        v_diff = abs(left_v_ratio - right_v_ratio)
        
        h_consistent = h_diff <= GazeParams.EYE_CONSISTENCY_THRESHOLD
        v_consistent = v_diff <= GazeParams.EYE_CONSISTENCY_THRESHOLD
        
        # 加权计算总一致性分数
        h_weight = GazeParams.HORIZONTAL_CONSISTENCY_WEIGHT
        v_weight = GazeParams.VERTICAL_CONSISTENCY_WEIGHT
        
        consistency_score = (h_weight * (1 - h_diff / max(1, abs(left_h_ratio) + abs(right_h_ratio))) + 
                           v_weight * (1 - v_diff / max(1, abs(left_v_ratio) + abs(right_v_ratio))))
        
        # 一致性条件：水平和垂直方向都符合阈值
        is_consistent = h_consistent and v_consistent
        
        return is_consistent, consistency_score
    
    def _determine_gaze_direction(self, left_h_ratio, left_v_ratio, left_gaze_3d,
                                 right_h_ratio, right_v_ratio, right_gaze_3d):
        """
        根据两只眼睛的虹膜位置确定视线方向，增强版
        
        Args:
            left_h_ratio: 左眼水平比例
            left_v_ratio: 左眼垂直比例
            left_gaze_3d: 左眼3D视线向量
            right_h_ratio: 右眼水平比例
            right_v_ratio: 右眼垂直比例
            right_gaze_3d: 右眼3D视线向量
            
        Returns:
            Tuple[str, float, float, float]: 方向，置信度，平均水平比例，平均垂直比例
        """
        # 首先检查左右眼一致性
        is_consistent, consistency_score = self._check_eye_consistency(
            left_h_ratio, left_v_ratio, right_h_ratio, right_v_ratio)
        
        # 平滑处理
        smooth_left_h = self._smooth_value(self.left_h_ratio_history, left_h_ratio)
        smooth_left_v = self._smooth_value(self.left_v_ratio_history, left_v_ratio)
        smooth_right_h = self._smooth_value(self.right_h_ratio_history, right_h_ratio)
        smooth_right_v = self._smooth_value(self.right_v_ratio_history, right_v_ratio)
        
        self.left_h_ratio_history.append(smooth_left_h)
        self.left_v_ratio_history.append(smooth_left_v)
        self.right_h_ratio_history.append(smooth_right_h)
        self.right_v_ratio_history.append(smooth_right_v)
        
        # 更新动态阈值
        h_ratios = list(self.left_h_ratio_history) + list(self.right_h_ratio_history)
        v_ratios = list(self.left_v_ratio_history) + list(self.right_v_ratio_history)
        if len(h_ratios) >= 3:
            self.h_ratio_variance = np.var(h_ratios) + 0.05
            self.v_ratio_variance = np.var(v_ratios) + 0.05
        
        # 基于一致性计算权重
        left_weight = GazeParams.LEFT_EYE_WEIGHT
        right_weight = GazeParams.RIGHT_EYE_WEIGHT
        
        # 如果一致性不好，确定哪只眼的数据更可靠
        if not is_consistent:
            # 使用简单的绝对值大小来判断更显著的信号
            if abs(smooth_left_v) >= abs(smooth_right_v):
                # 左眼垂直方向数据更可靠
                left_weight = 0.7
                right_weight = 0.3
            else:
                # 右眼垂直方向数据更可靠
                left_weight = 0.3
                right_weight = 0.7
        
        # 计算加权平均
        avg_h_ratio = left_weight * smooth_left_h + right_weight * smooth_right_h
        avg_v_ratio = left_weight * smooth_left_v + right_weight * smooth_right_v
        
        # 动态调整阈值 - 基于最近数据的变异性
        h_threshold = max(GazeParams.HORIZONTAL_RATIO_THRESHOLD, self.h_ratio_variance * 2)
        v_threshold = max(GazeParams.VERTICAL_RATIO_THRESHOLD, self.v_ratio_variance * 2)
        center_threshold = GazeParams.CENTER_THRESHOLD
        
        # 计算置信度 - 结合位置和一致性
        position_confidence = max(abs(avg_h_ratio), abs(avg_v_ratio))
        confidence = position_confidence * consistency_score
        
        # 判断视线方向 - 九个区域，使用更严格的上下阈值
        if abs(avg_h_ratio) <= center_threshold and abs(avg_v_ratio) <= center_threshold:
            return DIRECTION_CENTER, confidence, avg_h_ratio, avg_v_ratio
            
        # 垂直方向判断 - 使用更严格的向上阈值
        if avg_v_ratio <= GazeParams.UP_THRESHOLD_STRICTER:  # 向上看需要更明显的动作
            if avg_h_ratio >= h_threshold:
                return DIRECTION_UP_RIGHT, confidence, avg_h_ratio, avg_v_ratio
            elif avg_h_ratio <= -h_threshold:
                return DIRECTION_UP_LEFT, confidence, avg_h_ratio, avg_v_ratio
            else:
                return DIRECTION_UP, confidence, avg_h_ratio, avg_v_ratio
        elif avg_v_ratio >= GazeParams.DOWN_THRESHOLD_STRICTER:  # 下方
            if avg_h_ratio >= h_threshold:
                return DIRECTION_DOWN_RIGHT, confidence, avg_h_ratio, avg_v_ratio
            elif avg_h_ratio <= -h_threshold:
                return DIRECTION_DOWN_LEFT, confidence, avg_h_ratio, avg_v_ratio
            else:
                return DIRECTION_DOWN, confidence, avg_h_ratio, avg_v_ratio
        else:  # 中间高度
            if avg_h_ratio >= h_threshold:
                return DIRECTION_RIGHT, confidence, avg_h_ratio, avg_v_ratio
            elif avg_h_ratio <= -h_threshold:
                return DIRECTION_LEFT, confidence, avg_h_ratio, avg_v_ratio
            else:
                return DIRECTION_CENTER, confidence, avg_h_ratio, avg_v_ratio
    
    def _process_frame(self, frame: np.ndarray) -> GazeDirectionState:
        """
        处理图像帧，检测视线方向
        
        Args:
            frame: 输入图像帧
            
        Returns:
            GazeDirectionState: 视线方向状态
        """
        state = GazeDirectionState(frame=frame, timestamp=time.time())
        
        h, w, _ = frame.shape
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            
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
                
                # 增强的虹膜位置计算(包含3D视线向量)
                left_h_ratio, left_v_ratio, left_gaze_3d, left_pupil = self._calculate_iris_position(
                    left_eye_landmarks, left_iris_landmarks, frame)
                
                right_h_ratio, right_v_ratio, right_gaze_3d, right_pupil = self._calculate_iris_position(
                    right_eye_landmarks, right_iris_landmarks, frame)
                
                # 平滑处理瞳孔位置
                if left_pupil:
                    smooth_left_pupil = self._smooth_value(
                        self.left_pupil_center_history, left_pupil)
                    self.left_pupil_center_history.append(smooth_left_pupil)
                
                if right_pupil:
                    smooth_right_pupil = self._smooth_value(
                        self.right_pupil_center_history, right_pupil)
                    self.right_pupil_center_history.append(smooth_right_pupil)
                
                # 平滑3D视线向量
                if left_gaze_3d:
                    self.left_gaze_direction_history.append(left_gaze_3d)
                
                if right_gaze_3d:
                    self.right_gaze_direction_history.append(right_gaze_3d)
                
                # 整合增强版视线方向判断
                direction, confidence, avg_h_ratio, avg_v_ratio = self._determine_gaze_direction(
                    left_h_ratio, left_v_ratio, left_gaze_3d,
                    right_h_ratio, right_v_ratio, right_gaze_3d
                )

                # 更新状态
                state.detections["gaze_direction"]["direction"] = direction
                state.detections["gaze_direction"]["confidence"] = float(confidence)
                state.detections["gaze_direction"]["horizontal_ratio"] = float(avg_h_ratio)
                state.detections["gaze_direction"]["vertical_ratio"] = float(avg_v_ratio)
                state.detections["gaze_direction"]["left_eye"]["iris_position"] = (float(left_h_ratio), float(left_v_ratio))
                state.detections["gaze_direction"]["right_eye"]["iris_position"] = (float(right_h_ratio), float(right_v_ratio))
                state.detections["gaze_direction"]["left_eye"]["eye_landmarks"] = [(float(x), float(y)) for x, y in left_eye_landmarks]
                state.detections["gaze_direction"]["right_eye"]["eye_landmarks"] = [(float(x), float(y)) for x, y in right_eye_landmarks]
                state.detections["gaze_direction"]["face_detected"] = True
                
                # 调试可视化
                if self.debug:
                    # 绘制眼睛轮廓点
                    for point in left_eye_landmarks:
                        cv2.circle(frame, point, 1, (0, 255, 0), -1)
                    for point in right_eye_landmarks:
                        cv2.circle(frame, point, 1, (0, 255, 0), -1)
                    
                    # 绘制虹膜点
                    for point in left_iris_landmarks:
                        cv2.circle(frame, point, 1, (0, 0, 255), -1)
                    for point in right_iris_landmarks:
                        cv2.circle(frame, point, 1, (0, 0, 255), -1)
                    
                    # 绘制增强检测的瞳孔中心
                    if self.left_pupil_center_history:
                        left_pupil = self.left_pupil_center_history[-1]
                        cv2.circle(frame, (int(left_pupil[0]), int(left_pupil[1])), 3, (255, 0, 255), -1)
                    
                    if self.right_pupil_center_history:
                        right_pupil = self.right_pupil_center_history[-1]
                        cv2.circle(frame, (int(right_pupil[0]), int(right_pupil[1])), 3, (255, 0, 255), -1)
                    
                    # 显示视线方向和比例信息
                    direction_text = f"方向: {direction}, 置信度: {confidence:.2f}"
                    cv2.putText(frame, direction_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    ratio_text = f"水平: {avg_h_ratio:.2f}, 垂直: {avg_v_ratio:.2f}"
                    cv2.putText(frame, ratio_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # 显示动态阈值信息
                    thresh_text = f"H阈值: {h_threshold:.2f}, V阈值: {v_threshold:.2f}"
                    cv2.putText(frame, thresh_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    logger.debug(f"视线方向: {direction}, 置信度: {confidence:.2f}, 水平: {avg_h_ratio:.2f}, 垂直: {avg_v_ratio:.2f}")
        
        return state
    
    def start(self) -> int:
        """
        开始视线方向跟踪
        
        Returns:
            int: 错误码，0表示成功，其他值表示失败
        """
        result = super().start()
        if result != SUCCESS:
            logger.error(f"无法启动视线方向跟踪器: {result}")
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
                logger.error(f"关闭视线方向跟踪器时出错: {result}")
            
            return result
            
        except Exception as e:
            logger.error(f"关闭视线方向跟踪器失败: {str(e)}")
            return RUNTIME_ERROR
