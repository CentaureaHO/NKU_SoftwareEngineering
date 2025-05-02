import cv2
import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Union
import time
import os
import logging

from ..core import BaseModality, ModalityState
from ..core.error_codes import (
    SUCCESS, CAMERA_NOT_AVAILABLE, VIDEO_FILE_NOT_FOUND, 
    VIDEO_SOURCE_ERROR, FRAME_ACQUISITION_FAILED, RUNTIME_ERROR
)

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
        self.capture = None
        self.last_frame = None
        self.is_file_source = isinstance(source, str) and os.path.isfile(source)
        self.loop_video = True
    
    def initialize(self) -> int:
        """
        初始化视频源
        
        Returns:
            int: 错误码，0表示成功，其他值表示失败
        """
        try:
            # 尝试使用DirectShow后端打开摄像头
            if not self.is_file_source:
                self.capture = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
                self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            else:
                self.capture = cv2.VideoCapture(self.source)
            
            if not self.capture.isOpened():
                if not self.is_file_source:
                    self.capture = cv2.VideoCapture(self.source)
                    self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                    self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                    
                    if not self.capture.isOpened():
                        return CAMERA_NOT_AVAILABLE
                else:
                    return VIDEO_FILE_NOT_FOUND
                
            return SUCCESS
        except Exception as e:
            return VIDEO_SOURCE_ERROR
    
    def update(self) -> Optional[VisualState]:
        """
        读取一帧图像并处理
        
        Returns:
            Optional[VisualState]: 处理后的视觉状态
        """
        if not self._is_running or self.capture is None:
            logging.error(f"视频源未运行或未初始化 - is_running: {self._is_running}, capture: {self.capture is not None}")
            return None
        
        try:
            ret, frame = self.capture.read()
            
            if not ret:
                if self.is_file_source:
                    if self.loop_video:
                        logging.info("视频文件已结束，正在循环播放")
                        self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = self.capture.read()
                    else:
                        logging.info("视频文件已结束，不循环播放")
                
                if not ret:
                    logging.error("无法读取视频帧")
                    return None
            
            self.last_frame = frame
            
            state = self._process_frame(frame)
            self._last_state = state
            
            return state
            
        except Exception as e:
            logging.error(f"读取视频帧时出错: {str(e)}")
            return None
    
    def _process_frame(self, frame: np.ndarray) -> VisualState:
        """
        处理图像帧，子类需要重写此方法
        
        Args:
            frame: 输入图像帧
            
        Returns:
            VisualState: 处理后的视觉状态
        """
        return VisualState(frame=frame)
    
    def shutdown(self) -> int:
        """
        关闭视频源
        
        Returns:
            int: 错误码，0表示成功，其他值表示失败
        """
        try:
            if self.capture is not None:
                self.capture.release()
                self.capture = None
            
            self.last_frame = None
            return SUCCESS
        except Exception as e:
            return RUNTIME_ERROR
    
    def get_frame_size(self) -> Tuple[int, int]:
        """
        获取图像帧尺寸
        
        Returns:
            Tuple[int, int]: 宽度和高度
        """
        return (self.width, self.height)
    
    def set_loop_video(self, loop: bool) -> None:
        """
        设置是否循环播放视频文件
        
        Args:
            loop: 是否循环播放
        """
        self.loop_video = loop
