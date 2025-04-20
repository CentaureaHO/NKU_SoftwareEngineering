import cv2
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import math
import logging
import os

# 配置日志
logging.basicConfig(
    level=logging.DEBUG if os.environ.get('MODALITY_DEBUG', '0') == '1' else logging.INFO,  # 根据DEBUG环境变量设置日志级别
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='head_tracker.log',
    filemode='w'
)
logger = logging.getLogger('HeadTracker')

DEBUG = os.environ.get('MODALITY_DEBUG', '0') == '1'

import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

from .base_visual import BaseVisualModality, VisualState
from ..core.error_codes import (
    SUCCESS, MEDIAPIPE_INITIALIZATION_FAILED, RUNTIME_ERROR
)

class HeadState(VisualState):
    """头部状态类"""
    
    def __init__(self, frame=None, timestamp=None):
        super().__init__(frame, timestamp)
        self.detections = {
            "head_pose": {
                "pitch": 0.0,  # 俯仰角（点头）
                "yaw": 0.0,    # 偏航角（左右转头）
                "roll": 0.0,   # 翻滚角（头部倾斜）
                "detected": False,  # 是否检测到人脸
                "landmarks": []  # 关键点
            },
            "gaze": {
                "direction_x": 0.0,  # 视线水平方向 (-1: 左, 1: 右)
                "direction_y": 0.0,  # 视线垂直方向 (-1: 上, 1: 下)
                "confidence": 0.0,   # 视线方向估计的置信度
                "target": "unknown"  # 视线目标（道路、后视镜、仪表盘等）
            }
        }
    
    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        return result

class HeadTracker(BaseVisualModality):
    """
    头部跟踪器，跟踪用户的头部姿态、视线方向
    """
    
    def __init__(self, name: str = "head_tracker", source: int = 0, 
                 width: int = 640, height: int = 480,
                 min_detection_confidence: float = 0.5, 
                 min_tracking_confidence: float = 0.5,
                 debug: bool = DEBUG):
        """
        初始化头部跟踪器
        
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
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.debug = debug
        self.face_mesh = None
        
        # 设置MediaPipe参数
        self.mp_face_options = {
            'max_num_faces': 1,  # 只跟踪一个人脸
            'refine_landmarks': True,
            'min_detection_confidence': min_detection_confidence,
            'min_tracking_confidence': min_tracking_confidence
        }
        
        logger.info(f"头部跟踪器初始化完成，调试模式：{self.debug}")
    
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
            # 设置MediaPipe，提供图像尺寸以避免警告
            self.face_mesh = mp_face_mesh.FaceMesh(
                max_num_faces=self.mp_face_options['max_num_faces'],
                refine_landmarks=self.mp_face_options['refine_landmarks'],
                min_detection_confidence=self.mp_face_options['min_detection_confidence'],
                min_tracking_confidence=self.mp_face_options['min_tracking_confidence'],
            )
            logger.info("MediaPipe人脸网格初始化成功")
        except Exception as e:
            logger.error(f"MediaPipe初始化失败: {str(e)}")
            return MEDIAPIPE_INITIALIZATION_FAILED
            
        return SUCCESS
    
    def _process_frame(self, frame: np.ndarray) -> HeadState:
        """
        处理图像帧，检测头部状态
        
        Args:
            frame: 输入图像帧
            
        Returns:
            HeadState: 头部状态
        """
        state = HeadState(frame=frame)
        
        try:
            # 将BGR图像转为RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 为了提高性能，将图像标记为不可写
            rgb_frame.flags.writeable = False
            
            # 处理图像
            results = self.face_mesh.process(rgb_frame)
            
            # 标记图像为可写
            rgb_frame.flags.writeable = True
            
            if results.multi_face_landmarks:
                logger.debug("检测到人脸")
                face_landmarks = results.multi_face_landmarks[0]
                
                # 计算头部姿态
                pitch, yaw, roll = self._calculate_head_pose(face_landmarks, frame.shape)
                
                # 估计视线方向
                gaze_x, gaze_y, gaze_conf, target = self._estimate_gaze_direction(face_landmarks, frame.shape)
                
                # 更新状态
                state.detections["head_pose"]["pitch"] = pitch
                state.detections["head_pose"]["yaw"] = yaw
                state.detections["head_pose"]["roll"] = roll
                state.detections["head_pose"]["detected"] = True
                
                # 存储关键点 (只存储必要的关键点，减少内存占用)
                landmarks = []
                essential_landmarks = [1, 33, 133, 159, 145, 263, 362, 374, 386, 473, 468]  # 基本面部特征点
                for idx in essential_landmarks:
                    if idx < len(face_landmarks.landmark):
                        landmark = face_landmarks.landmark[idx]
                        x = int(landmark.x * frame.shape[1])
                        y = int(landmark.y * frame.shape[0])
                        z = landmark.z
                        landmarks.append((x, y, z))
                
                state.detections["head_pose"]["landmarks"] = landmarks
                
                # 更新视线信息
                state.detections["gaze"]["direction_x"] = gaze_x
                state.detections["gaze"]["direction_y"] = gaze_y
                state.detections["gaze"]["confidence"] = gaze_conf
                state.detections["gaze"]["target"] = target
                
            else:
                logger.debug("未检测到人脸")
        
        except Exception as e:
            logger.error(f"处理帧时出错: {str(e)}")
        
        return state
    
    def _calculate_head_pose(self, face_landmarks, img_shape) -> Tuple[float, float, float]:
        """
        计算头部姿态
        
        Args:
            face_landmarks: MediaPipe面部关键点
            img_shape: 图像尺寸
            
        Returns:
            Tuple[float, float, float]: 俯仰角、偏航角、翻滚角（度数）
        """
        img_h, img_w, _ = img_shape
        
        try:
            # 提取特征点
            # 鼻尖
            nose = face_landmarks.landmark[1]
            nose_x = int(nose.x * img_w)
            nose_y = int(nose.y * img_h)
            
            # 下巴
            chin = face_landmarks.landmark[152]
            chin_x = int(chin.x * img_w)
            chin_y = int(chin.y * img_h)
            
            # 左眼和右眼中心
            left_eye = face_landmarks.landmark[159]
            left_eye_x = int(left_eye.x * img_w)
            left_eye_y = int(left_eye.y * img_h)
            
            right_eye = face_landmarks.landmark[386]
            right_eye_x = int(right_eye.x * img_w)
            right_eye_y = int(right_eye.y * img_h)
            
            # 计算头部姿态
            # 偏航角 (左右转动) - 基于眼睛的水平位置差异
            eye_dx = right_eye_x - left_eye_x
            if abs(eye_dx) < 0.001:
                eye_dx = 0.001  # 避免除以零
            eye_dy = right_eye_y - left_eye_y
            eye_angle = math.atan2(eye_dy, eye_dx)
            roll = math.degrees(eye_angle)
            
            # 俯仰角 (上下点头) - 基于鼻子和下巴的垂直关系
            nose_chin_dy = chin_y - nose_y
            pitch = (nose_chin_dy / img_h) * 90.0  # 归一化到大约 -45 到 45 度
            
            # 偏航角 (左右摇头) - 基于两眼中心与鼻子的关系
            eye_center_x = (left_eye_x + right_eye_x) / 2
            yaw = ((nose_x - eye_center_x) / img_w) * 90.0  # 归一化到大约 -45 到 45 度
            
            logger.debug(f"头部姿态: 俯仰={pitch:.1f}, 偏航={yaw:.1f}, 翻滚={roll:.1f}")
            return pitch, yaw, roll
            
        except Exception as e:
            logger.error(f"计算头部姿态时出错: {str(e)}")
            return 0.0, 0.0, 0.0
    
    def _estimate_gaze_direction(self, face_landmarks, img_shape) -> Tuple[float, float, float, str]:
        """
        估计视线方向
        
        Args:
            face_landmarks: MediaPipe面部关键点
            img_shape: 图像尺寸
            
        Returns:
            Tuple[float, float, float, str]: x方向, y方向, 置信度, 视线目标
        """
        img_h, img_w, _ = img_shape
        
        try:
            # 提取特征点 - 眼睛关键点
            # 左眼
            left_iris_center = face_landmarks.landmark[468]  # 虹膜中心
            left_eye_inner = face_landmarks.landmark[362]
            left_eye_outer = face_landmarks.landmark[263]
            left_eye_top = face_landmarks.landmark[386]
            left_eye_bottom = face_landmarks.landmark[374]
            
            # 右眼
            right_iris_center = face_landmarks.landmark[473]  # 虹膜中心
            right_eye_inner = face_landmarks.landmark[133]
            right_eye_outer = face_landmarks.landmark[33]
            right_eye_top = face_landmarks.landmark[159]
            right_eye_bottom = face_landmarks.landmark[145]
            
            # 计算左眼虹膜相对位置
            left_eye_width = abs(left_eye_outer.x - left_eye_inner.x)
            if left_eye_width < 0.001:
                left_eye_width = 0.001  # 避免除以零
            left_iris_rel_x = (left_iris_center.x - left_eye_inner.x) / left_eye_width - 0.5
            
            left_eye_height = abs(left_eye_top.y - left_eye_bottom.y)
            if left_eye_height < 0.001:
                left_eye_height = 0.001  # 避免除以零
            left_iris_rel_y = (left_iris_center.y - left_eye_top.y) / left_eye_height - 0.5
            
            # 计算右眼虹膜相对位置
            right_eye_width = abs(right_eye_outer.x - right_eye_inner.x)
            if right_eye_width < 0.001:
                right_eye_width = 0.001  # 避免除以零
            right_iris_rel_x = (right_iris_center.x - right_eye_inner.x) / right_eye_width - 0.5
            
            right_eye_height = abs(right_eye_top.y - right_eye_bottom.y)
            if right_eye_height < 0.001:
                right_eye_height = 0.001  # 避免除以零
            right_iris_rel_y = (right_iris_center.y - right_eye_top.y) / right_eye_height - 0.5
            
            # 平均两只眼睛的结果
            gaze_x = (left_iris_rel_x + right_iris_rel_x) / 2 * 2  # 归一化到 -1 到 1
            gaze_y = (left_iris_rel_y + right_iris_rel_y) / 2 * 2  # 归一化到 -1 到 1
            
            # 置信度 - 基于两眼一致性
            x_diff = abs(left_iris_rel_x - right_iris_rel_x)
            y_diff = abs(left_iris_rel_y - right_iris_rel_y)
            confidence = max(0.0, 1.0 - (x_diff + y_diff))
            
            # 确定视线目标
            target = "unknown"
            
            # 这些阈值和区域划分需要根据实际场景调整
            if abs(gaze_x) < 0.2 and abs(gaze_y) < 0.2:
                target = "center"  # 看中心
            elif gaze_y < -0.3:
                target = "up"  # 看上方
            elif gaze_y > 0.3:
                target = "down"  # 看下方
            elif gaze_x > 0.3:
                target = "right"  # 看右侧
            elif gaze_x < -0.3:
                target = "left"  # 看左侧
            
            logger.debug(f"视线方向: x={gaze_x:.2f}, y={gaze_y:.2f}, 目标={target}")
            return gaze_x, gaze_y, confidence, target
            
        except Exception as e:
            logger.error(f"估计视线方向时出错: {str(e)}")
            return 0.0, 0.0, 0.0, "unknown"
    
    def shutdown(self) -> int:
        """
        关闭头部跟踪器资源
        
        Returns:
            int: 错误码，0表示成功，其他值表示失败
        """
        try:
            if self.face_mesh:
                self.face_mesh.close()
                self.face_mesh = None
            
            result = super().shutdown()
            return result
        except Exception as e:
            logger.error(f"关闭头部跟踪器失败: {str(e)}")
            return RUNTIME_ERROR
    
    # 替换原来的release方法
    def release(self) -> int:
        """
        释放资源（已弃用，请使用shutdown）
        
        Returns:
            int: 错误码，0表示成功，其他值表示失败
        """
        logger.info("release方法已弃用，请使用shutdown")
        return self.shutdown()
