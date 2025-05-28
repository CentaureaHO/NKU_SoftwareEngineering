import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from utils.camera_manager import CameraManager, get_camera_manager

from ..core import BaseModality, ModalityState
from ..core.error_codes import (CAMERA_NOT_AVAILABLE, FRAME_ACQUISITION_FAILED,
                                RUNTIME_ERROR, SUCCESS, VIDEO_FILE_NOT_FOUND,
                                VIDEO_SOURCE_ERROR)


class VisualState(ModalityState):
    """
    视觉模态状态类，存储视觉感知的结果
    """

    def __init__(self, frame: Optional[np.ndarray] = None, timestamp: float = None):
        super().__init__(timestamp)
        self.frame = frame                    # 原始图像帧
        self.detections: Dict[str, Any] = {}  # 检测结果

    def to_dict(self) -> Dict[str, Any]:
        """将状态转换为字典格式"""
        result = super().to_dict()
        result["frame_shape"] = self.frame.shape if self.frame is not None else None
        result["detections"] = self.detections
        return result


class BaseVisualModality(BaseModality):
    """
    视觉模态基类，提供视觉处理的通用功能
    """

    def __init__(self, name: str, source: Union[int, str] = 0, width: int = 640, height: int = 480):
        """
        Args:
            name: 模态名称
            source: 视频源，可以是摄像头ID（整数）或视频文件路径（字符串）
            width: 图像宽度
            height: 图像高度
        """
        super().__init__(name)
        self.source = source
        self.width = width
        self.height = height
        self.camera_manager: CameraManager = get_camera_manager()
        self.last_frame: Optional[np.ndarray] = None
        self.loop_video: bool = True

    def initialize(self) -> int:
        """
        初始化视频源

        Returns:
            int: 错误码，0表示成功，其他值表示失败
        """
        try:
            init_status = self.camera_manager.initialize_camera(
                source=self.source,
                width=self.width,
                height=self.height,
                loop_video=self.loop_video
            )
            if init_status != SUCCESS:
                logging.error(f"CameraManager failed to initialize for source {
                              self.source}. Error code: {init_status}")
                return init_status

            props = self.camera_manager.get_properties()
            if props.get('is_opened'):
                self.width = int(props.get('width', self.width))
                self.height = int(props.get('height', self.height))
                logging.info(f"Camera initialized via CameraManager. Effective resolution: {
                             self.width}x{self.height}")

            return SUCCESS
        except Exception as e:
            logging.error(f"Exception during BaseVisualModality initialize via CameraManager: {
                          str(e)}", exc_info=True)
            return VIDEO_SOURCE_ERROR

    def update(self) -> Optional[VisualState]:
        """
        读取一帧图像并处理

        Returns:
            Optional[VisualState]: 处理后的视觉状态
        """
        if not self._is_running:
            logging.warning(
                f"{self.name} modality is not running, cannot update.")
            return None

        if not self.camera_manager.is_running():
            logging.error(f"CameraManager is not running or not initialized for source {
                          self.source}. Cannot read frame.")
            return None

        try:
            ret, frame = self.camera_manager.read_frame()

            if not ret:
                logging.warning(
                    f"Failed to read frame from CameraManager for source {self.source}.")
                return None

            self.last_frame = frame.copy()

            state = self._process_frame(frame)
            self._last_state = state

            return state

        except Exception as e:
            logging.error(f"Error processing frame in BaseVisualModality update: {
                          str(e)}", exc_info=True)
            return None

    def _process_frame(self, frame: np.ndarray) -> VisualState:
        """
        处理图像帧，子类需要重写此方法

        Args:
            frame: 输入图像帧

        Returns:
            VisualState: 处理后的视觉状态
        """
        return VisualState(frame=frame, timestamp=time.time())

    def shutdown(self) -> int:
        """
        关闭视频源

        Returns:
            int: 错误码，0表示成功，其他值表示失败
        """
        try:
            logging.info(f"BaseVisualModality '{self.name}' shutdown. CameraManager for source '{
                         self.source}' remains active if shared.")
            self.last_frame = None
        except Exception as e:
            logging.error(f"Error during BaseVisualModality shutdown: {
                          str(e)}", exc_info=True)
            return RUNTIME_ERROR

    def get_frame_size(self) -> Tuple[int, int]:
        """
        获取图像帧尺寸

        Returns:
            Tuple[int, int]: 宽度和高度
        """
        if self.camera_manager.is_running():
            props = self.camera_manager.get_properties()
            cam_width = props.get('width')
            cam_height = props.get('height')
            if cam_width is not None and cam_height is not None:
                return (int(cam_width), int(cam_height))
        return (self.width, self.height)

    def set_loop_video(self, loop: bool) -> None:
        """
        设置是否循环播放视频文件

        Args:
            loop: 是否循环播放
        """
        self.loop_video = loop
        if self.camera_manager:
            self.camera_manager.set_loop_video(loop)
            logging.info(f"BaseVisualModality '{self.name}' loop_video set to {
                         loop}. Updated CameraManager.")
