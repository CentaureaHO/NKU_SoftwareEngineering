import cv2
import numpy as np
from typing import Dict, Any
import logging
import os
from collections import deque
import time
import threading
import sys

logging.basicConfig(
    level=logging.DEBUG if os.environ.get('MODALITY_DEBUG', '0') == '1' else logging.INFO,
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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Modality.visual.base_visual import BaseVisualModality
from Modality.visual.head_pose_common import HeadPoseParams, HeadPoseState, euclidean_dist, rotation_matrix_to_angles
from Modality.core.error_codes import (
    SUCCESS, MEDIAPIPE_INITIALIZATION_FAILED, RUNTIME_ERROR
)


class HeadPoseTrackerGeom(BaseVisualModality):
    """
    几何方法实现点头和摇头检测，通过独立线程运行
    """
    
    def __init__(self, name: str = "head_pose_tracker_geom", source: int = 0, 
                 width: int = 640, height: int = 480,
                 min_detection_confidence: float = 0.5, 
                 min_tracking_confidence: float = 0.5,
                 debug: bool = DEBUG):
        """
        初始化基于几何方法的头部姿态跟踪器
        
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
        
        # 面部关键点3D坐标
        self.face_3d_coords = np.array([
            [285, 528, 200],  # 鼻子
            [285, 371, 152],  # 鼻子下方
            [197, 574, 128],  # 左脸边缘
            [173, 425, 108],  # 左眼附近
            [360, 574, 128],  # 右脸边缘
            [391, 425, 108]   # 右眼附近
        ], dtype=np.float64)
        
        # 历史记录队列，用于平滑检测结果
        self.nose_chin_distances = deque(maxlen=HeadPoseParams.HISTORY_LEN)
        self.left_cheek_widths = deque(maxlen=HeadPoseParams.HISTORY_LEN)
        self.right_cheek_widths = deque(maxlen=HeadPoseParams.HISTORY_LEN)
        self.pitch_angles = deque(maxlen=HeadPoseParams.HISTORY_LEN)
        self.yaw_angles = deque(maxlen=HeadPoseParams.HISTORY_LEN)
        
        # 原始动作检测结果队列，用于voting出当前状态
        self.raw_nod_results = deque(maxlen=HeadPoseParams.HISTORY_LEN)
        self.raw_shake_results = deque(maxlen=HeadPoseParams.HISTORY_LEN)
        
        self._processing_thread = None
        self._stop_event = threading.Event()
        
        self._latest_state = HeadPoseState()
        self._state_lock = threading.Lock()
        
        self._last_status_update = time.time()
        
        # 状态缓存，解决_process_frame中间隔时间内的状态不一致问题
        self._current_is_nodding = False
        self._current_is_shaking = False
        self._current_nod_confidence = 0.0
        self._current_shake_confidence = 0.0
        self._current_status = HeadPoseParams.STATUS_STATIONARY
        
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
            # 设置MediaPipe
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
        
        nose_point = None
        chin_point = None
        left_ear_point = None
        left_face_point = None
        right_ear_point = None
        right_face_point = None
        
        # 原始检测结果
        current_nod_detected = False
        current_shake_detected = False
        
        if results.multi_face_landmarks:
            face_coordination_in_image = []
            
            for face_landmarks in results.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    x, y = int(lm.x * w), int(lm.y * h)
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
                
                landmarks_list = []
                for i, landmark in enumerate(face_landmarks.landmark):
                    if i in HeadPoseParams.ESSENTIAL_LANDMARKS:
                        x, y, z = landmark.x, landmark.y, landmark.z
                        landmarks_list.append((x, y, z))
                
                state.detections["head_pose"]["landmarks"] = landmarks_list
                state.detections["head_pose"]["detected"] = True
                
                if len(face_coordination_in_image) == 6:
                    face_coordination_in_image = np.array(face_coordination_in_image, dtype=np.float64)
                    
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
                        
                        rotation_matrix, _ = cv2.Rodrigues(rotation_vec)
                        angles = rotation_matrix_to_angles(rotation_matrix)
                        
                        current_pitch = angles[0]  # 俯仰角（点头）
                        current_yaw = angles[1]    # 偏航角（左右转头）
                        current_roll = angles[2]   # 翻滚角（头部倾斜）
                        
                        state.detections["head_pose"]["pitch"] = float(current_pitch)
                        state.detections["head_pose"]["yaw"] = float(current_yaw)
                        state.detections["head_pose"]["roll"] = float(current_roll)
                        
                        self.pitch_angles.append(current_pitch)
                        self.yaw_angles.append(current_yaw)
                        
                        if nose_point and chin_point and left_ear_point and left_face_point and right_ear_point and right_face_point:
                            nose_chin_dist = euclidean_dist(nose_point, chin_point)
                    
                            left_cheek_width = euclidean_dist(left_ear_point, left_face_point)
                            right_cheek_width = euclidean_dist(right_ear_point, right_face_point)
                            
                            self.nose_chin_distances.append(nose_chin_dist)
                            self.left_cheek_widths.append(left_cheek_width)
                            self.right_cheek_widths.append(right_cheek_width)
                            
                            current_nod_detected = self._detect_nod(
                                list(self.pitch_angles), 
                                list(self.nose_chin_distances)
                            )
                            
                            current_shake_detected = self._detect_shake(
                                list(self.yaw_angles), 
                                list(self.left_cheek_widths), 
                                list(self.right_cheek_widths)
                            )
                        
                            self.raw_nod_results.append(1 if current_nod_detected else 0)
                            self.raw_shake_results.append(1 if current_shake_detected else 0)
                            
                            is_nodding, is_shaking, nod_ratio, shake_ratio = self._get_movement_status(
                                self.raw_nod_results, 
                                self.raw_shake_results
                            )
                            
                            current_time = time.time()
                            if current_time - self._last_status_update >= HeadPoseParams.STATUS_UPDATE_INTERVAL:
                                self._last_status_update = current_time
                                
                                self._current_is_nodding = is_nodding
                                self._current_is_shaking = is_shaking
                                self._current_nod_confidence = float(nod_ratio)
                                self._current_shake_confidence = float(shake_ratio)
                                
                                if is_nodding:
                                    self._current_status = HeadPoseParams.STATUS_NODDING
                                elif is_shaking:
                                    self._current_status = HeadPoseParams.STATUS_SHAKING
                                else:
                                    self._current_status = HeadPoseParams.STATUS_STATIONARY
                            
                            state.detections["head_movement"]["is_nodding"] = self._current_is_nodding
                            state.detections["head_movement"]["is_shaking"] = self._current_is_shaking
                            state.detections["head_movement"]["nod_confidence"] = self._current_nod_confidence
                            state.detections["head_movement"]["shake_confidence"] = self._current_shake_confidence
                            state.detections["head_movement"]["status"] = self._current_status
                    
                    except Exception as e:
                        logger.error(f"计算头部姿态时出错: {str(e)}")
        
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
    
    def _detect_nod(self, pitch_values, nose_chin_values):
        """
        检测点头动作
        :param pitch_values: pitch角度历史数据
        :param nose_chin_values: 鼻子到下巴距离历史数据
        :return: 是否检测到点头动作
        """
        if len(pitch_values) < HeadPoseParams.MIN_HISTORY_DATA or len(nose_chin_values) < HeadPoseParams.MIN_HISTORY_DATA:
            return False
            
        # 条件1: 利用pitch角度变化检测点头
        pitch_std = np.std(pitch_values)  # 计算角度标准差
        pitch_range = max(pitch_values) - min(pitch_values)  # 计算角度范围
        
        if pitch_range < HeadPoseParams.STABLE_THRESHOLD:
            return False  # 如果角度范围小于阈值，认为头部静止
        
        # 条件2: 利用鼻子到下巴距离变化检测点头
        # 计算鼻子到下巴距离的变化率
        nose_chin_diffs = []
        for i in range(1, len(nose_chin_values)):
            prev = nose_chin_values[i-1]
            curr = nose_chin_values[i]
            if prev > 0:
                diff = (curr - prev) / prev
                nose_chin_diffs.append(diff)
        
        if not nose_chin_diffs:
            return False
            
        nc_std = np.std(nose_chin_diffs) * 100
        
        # 检测点头的鼻子-下巴距离变化模式: 先减后增 || 先增后减
        has_increase = any(d > HeadPoseParams.NC_CHANGE_THRESHOLD for d in nose_chin_diffs)  # 有明显增加
        has_decrease = any(d < -HeadPoseParams.NC_CHANGE_THRESHOLD for d in nose_chin_diffs)  # 有明显减少
        
        # 满足任一条件即可认为在点头
        is_nodding = (pitch_range > HeadPoseParams.NOD_THRESHOLD and pitch_std > HeadPoseParams.NOD_THRESHOLD/2) or \
                     (nc_std > 0.8 and has_increase and has_decrease)
                      
        return is_nodding
    
    def _detect_shake(self, yaw_values, left_values, right_values):
        """
        检测摇头动作
        :param yaw_values: yaw角度历史数据
        :param left_values: 左脸颊宽度历史数据
        :param right_values: 右脸颊宽度历史数据
        :return: 是否检测到摇头动作
        """
        if len(yaw_values) < HeadPoseParams.MIN_HISTORY_DATA or len(left_values) < HeadPoseParams.MIN_HISTORY_DATA or len(right_values) < HeadPoseParams.MIN_HISTORY_DATA:
            return False
            
        # 条件1: 利用yaw角度变化检测摇头
        yaw_std = np.std(yaw_values)  # 计算角度标准差
        yaw_range = max(yaw_values) - min(yaw_values)  # 计算角度范围
        
        if yaw_range < HeadPoseParams.STABLE_THRESHOLD:
            return False  # 如果角度范围小于阈值，认为头部静止
            
        # 条件2: 分析左右脸颊宽度的变化关系
        left_range = max(left_values) - min(left_values)
        right_range = max(right_values) - min(right_values)
        
        width_ratios = []
        for l, r in zip(left_values, right_values):
            if r > 0:
                width_ratios.append(l / r)
        
        # 检测是否有左右变化的交替模式 (摇头时每侧均为交替变化)
        left_derivatives = np.diff(left_values)
        right_derivatives = np.diff(right_values)
        
        # 计算变化方向的相关性 (摇头时通常为负相关)
        direction_correlation = np.corrcoef(left_derivatives, right_derivatives)[0, 1] if len(left_derivatives) > 1 else 0
        
        # 满足任一条件即可认为在摇头
        is_shaking = (yaw_range > HeadPoseParams.SHAKE_THRESHOLD) or \
                    (left_range > 4.0 and right_range > 4.0 and direction_correlation < -0.2)
        
        return is_shaking
    
    def _get_movement_status(self, raw_nods, raw_shakes):
        """
        根据原始检测结果确定最终动作状态
        :param raw_nods: 原始点头检测结果
        :param raw_shakes: 原始摇头检测结果
        :return: (is_nodding, is_shaking, nod_ratio, shake_ratio) 当前是否在点头和摇头及其置信度
        """
        if len(raw_nods) < HeadPoseParams.MIN_HISTORY_DATA or len(raw_shakes) < HeadPoseParams.MIN_HISTORY_DATA:
            return False, False, 0.0, 0.0
            
        nod_ratio = sum(raw_nods) / len(raw_nods)
        shake_ratio = sum(raw_shakes) / len(raw_shakes)
        is_nodding = nod_ratio >= HeadPoseParams.NOD_RATIO_THRESHOLD
        is_shaking = shake_ratio >= HeadPoseParams.SHAKE_RATIO_THRESHOLD
        
        # 如果两者都检测到，取较高的置信度作为当前状态
        if is_nodding and is_shaking:
            if nod_ratio > shake_ratio:
                is_shaking = False
            else:
                is_nodding = False
                
        return is_nodding, is_shaking, nod_ratio, shake_ratio
