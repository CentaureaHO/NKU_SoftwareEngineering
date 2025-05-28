import cv2
import threading
import os
import logging
import numpy as np
from typing import Optional, Union, Tuple
import time
import queue

try:
    from Modality.core.error_codes import (
        SUCCESS, CAMERA_NOT_AVAILABLE, VIDEO_FILE_NOT_FOUND,
        VIDEO_SOURCE_ERROR, FRAME_ACQUISITION_FAILED
    )
except ImportError:
    logging.warning("Failed to import error codes from Modality.core. Using fallback definitions.")
    SUCCESS = 0
    CAMERA_NOT_AVAILABLE = 101
    VIDEO_FILE_NOT_FOUND = 102
    VIDEO_SOURCE_ERROR = 103
    FRAME_ACQUISITION_FAILED = 104

logger = logging.getLogger('CameraManager')

class CameraManager:
    _instance = None
    _class_lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._class_lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized_once') and self._initialized_once:
            return
        
        self.capture: Optional[cv2.VideoCapture] = None
        self.source: Union[int, str] = 0
        self.width: int = 640
        self.height: int = 480
        self.is_file_source: bool = False
        self.loop_video: bool = True
        self._is_initialized: bool = False
        self._initialized_params: Optional[Tuple[Union[int, str], int, int, bool]] = None

        self._frame_buffer = queue.Queue(maxsize=10)
        self._latest_frame = None
        self._capture_thread = None
        self._stop_capture = threading.Event()
        
        self._operation_lock = threading.Lock()
        self._initialized_once = True
        logger.info("CameraManager instance created/accessed.")

    def _capture_frames(self):
        """持续捕获帧并放入缓冲区的线程函数"""
        logger.info(f"Frame capture thread started for source: {self.source}")
        retry_count = 0
        max_retries = 5
        retry_delay = 1.0
        
        while not self._stop_capture.is_set():
            if not self.capture or not self.capture.isOpened():
                logger.warning("Camera not opened in capture thread. Attempting to reconnect...")
                with self._operation_lock:
                    if self.is_file_source:
                        self.capture = cv2.VideoCapture(self.source)
                    else:
                        self.capture = cv2.VideoCapture(self.source)
                
                    if not self.capture.isOpened():
                        retry_count += 1
                        if retry_count > max_retries:
                            logger.error(f"Failed to reconnect to camera after {max_retries} attempts")
                            break
                        time.sleep(retry_delay)
                        continue
                    else:
                        retry_count = 0
                        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                        logger.info("Successfully reconnected to camera")
            
            try:
                ret, frame = self.capture.read()
                if not ret:
                    if self.is_file_source and self.loop_video:
                        logger.info(f"Video file '{self.source}' ended. Looping...")
                        self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        logger.warning("Failed to read frame. Retrying...")
                        retry_count += 1
                        if retry_count > max_retries:
                            logger.error(f"Failed to read frame after {max_retries} attempts")
                            break
                        time.sleep(retry_delay)
                        continue
                
                retry_count = 0
                
                with self._operation_lock:
                    self._latest_frame = frame.copy()
                
                try:
                    if self._frame_buffer.full():
                        try:
                            self._frame_buffer.get_nowait()
                        except queue.Empty:
                            pass
                    self._frame_buffer.put_nowait(frame)
                except:
                    pass

                time.sleep(0.01)
            
            except Exception as e:
                logger.error(f"Error in capture thread: {str(e)}", exc_info=True)
                retry_count += 1
                if retry_count > max_retries:
                    logger.error(f"Too many errors in capture thread. Exiting.")
                    break
                time.sleep(retry_delay)
        
        with self._operation_lock:
            if self.capture:
                self.capture.release()
                self.capture = None
        logger.info("Frame capture thread stopped")

    def initialize_camera(self, source: Union[int, str] = 0, width: int = 640, height: int = 480, loop_video: bool = True) -> int:
        with self._operation_lock:
            current_params = (source, width, height, loop_video)
            
            if self._is_initialized and self._initialized_params == current_params and self._capture_thread and self._capture_thread.is_alive():
                logger.info(f"Camera already initialized and running with same parameters: {current_params}")
                return SUCCESS
            
            self._stop_existing_thread()
            
            self.source = source
            self.width = width
            self.height = height
            self.is_file_source = isinstance(source, str)
            self.loop_video = loop_video
            
            logger.info(f"Initializing camera: source={self.source}, width={self.width}, height={self.height}, is_file={self.is_file_source}, loop={self.loop_video}")
            
            try:
                if self.is_file_source:
                    if not os.path.isfile(self.source):
                        logger.error(f"Video file not found: {self.source}")
                        self._is_initialized = False
                        return VIDEO_FILE_NOT_FOUND
                    self.capture = cv2.VideoCapture(self.source)
                else:
                    self.capture = cv2.VideoCapture(self.source)
                
                if not self.capture or not self.capture.isOpened():
                    logger.error(f"Failed to open video source: {self.source}")
                    self.capture = None
                    self._is_initialized = False
                    return CAMERA_NOT_AVAILABLE if not self.is_file_source else VIDEO_FILE_NOT_FOUND
                
                self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                
                actual_width = self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)
                actual_height = self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
                logger.info(f"Camera opened. Requested: {self.width}x{self.height}, Actual: {int(actual_width)}x{int(actual_height)}")
                
                while not self._frame_buffer.empty():
                    try:
                        self._frame_buffer.get_nowait()
                    except:
                        pass
                
                self._stop_capture.clear()
                self._capture_thread = threading.Thread(target=self._capture_frames, daemon=True)
                self._capture_thread.start()
                
                self._is_initialized = True
                self._initialized_params = current_params
                logger.info("Camera initialized successfully.")
                return SUCCESS
            
            except Exception as e:
                logger.error(f"Error initializing camera: {str(e)}", exc_info=True)
                self._release_capture_internal()
                return VIDEO_SOURCE_ERROR

    def _stop_existing_thread(self):
        """停止现有的捕获线程"""
        if self._capture_thread and self._capture_thread.is_alive():
            logger.info("Stopping existing capture thread...")
            self._stop_capture.set()
            self._capture_thread.join(timeout=2.0)
            if self._capture_thread.is_alive():
                logger.warning("Capture thread did not stop in time - continuing anyway")
        
        if self.capture:
            try:
                self.capture.release()
            except:
                pass
            self.capture = None
        
        self._capture_thread = None
        self._is_initialized = False

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """读取一帧。首先尝试从缓冲区获取，如果没有可用帧则返回最新帧或失败"""
        if not self._is_initialized:
            return False, None

        try:
            frame = self._frame_buffer.get_nowait()
            return True, frame
        except queue.Empty:
            pass

        with self._operation_lock:
            if self._latest_frame is not None:
                return True, self._latest_frame.copy()
        
        return False, None

    def _release_capture_internal(self):
        """内部方法：释放摄像头资源"""
        self._stop_existing_thread()
        self._latest_frame = None
        while not self._frame_buffer.empty():
            try:
                self._frame_buffer.get_nowait()
            except:
                pass

    def release_camera(self) -> int:
        """释放摄像头资源"""
        with self._operation_lock:
            logger.info(f"Explicitly releasing camera (source: {self.source}).")
            self._release_capture_internal()
            self._initialized_params = None
            logger.info("Camera released.")
        return SUCCESS

    def is_running(self) -> bool:
        """检查摄像头是否在运行"""
        return (self._is_initialized and 
                self._capture_thread is not None and 
                self._capture_thread.is_alive())

    def get_properties(self) -> dict:
        """获取摄像头属性"""
        with self._operation_lock:
            if self.capture and self.is_running():
                return {
                    "source": self.source,
                    "width": self.capture.get(cv2.CAP_PROP_FRAME_WIDTH),
                    "height": self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT),
                    "fps": self.capture.get(cv2.CAP_PROP_FPS),
                    "is_file_source": self.is_file_source,
                    "loop_video": self.loop_video,
                    "is_opened": True,
                    "backend_name": self.capture.getBackendName() if hasattr(self.capture, 'getBackendName') else 'N/A',
                    "buffer_size": self._frame_buffer.qsize()
                }
        return {
            "source": self.source,
            "width": self.width,
            "height": self.height,
            "fps": None,
            "is_file_source": self.is_file_source,
            "loop_video": self.loop_video,
            "is_opened": False,
            "backend_name": 'N/A',
            "buffer_size": 0
        }

    def set_loop_video(self, loop: bool):
        """设置视频循环播放"""
        with self._operation_lock:
            if self.loop_video != loop:
                self.loop_video = loop
                logger.info(f"Video loop behavior set to: {self.loop_video} for source {self.source}")
                if self._is_initialized and self._initialized_params:
                    self._initialized_params = (self._initialized_params[0], self._initialized_params[1], self._initialized_params[2], self.loop_video)


def get_camera_manager():
    return CameraManager()
