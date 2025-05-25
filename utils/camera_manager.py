import cv2
import threading
import os
import logging
import numpy as np
from typing import Optional, Union, Tuple

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
        
        self._operation_lock = threading.Lock()
        self._initialized_once = True
        logger.info("CameraManager instance created/accessed.")

    def initialize_camera(self, source: Union[int, str] = 0, width: int = 640, height: int = 480, loop_video: bool = True) -> int:
        with self._operation_lock:
            current_params = (source, width, height, loop_video)
            if self._is_initialized and self._initialized_params == current_params:
                if self.capture and self.capture.isOpened():
                    logger.info(f"Camera already initialized and open with same parameters: {current_params}")
                    return SUCCESS
                else:
                    logger.warning(f"Camera was initialized with same parameters but not open. Re-initializing: {current_params}")
            elif self._is_initialized and self._initialized_params != current_params:
                logger.info(f"Re-initializing camera with new parameters. Old: {self._initialized_params}, New: {current_params}")
                self._release_capture_internal()

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
                    self.capture = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
                    if not self.capture.isOpened():
                        logger.warning(f"Failed to open camera with CAP_DSHOW (source: {self.source}). Trying default backend.")
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

                self._is_initialized = True
                self._initialized_params = current_params
                logger.info("Camera initialized successfully.")
                return SUCCESS
            except Exception as e:
                logger.error(f"Error initializing camera: {str(e)}", exc_info=True)
                self._release_capture_internal()
                return VIDEO_SOURCE_ERROR

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self._operation_lock:
            if not self._is_initialized or not self.capture or not self.capture.isOpened():
                return False, None

            try:
                ret, frame = self.capture.read()
                if not ret:
                    if self.is_file_source and self.loop_video:
                        logger.info(f"Video file '{self.source}' ended. Looping...")
                        self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = self.capture.read()
                        if not ret:
                            logger.error(f"Failed to read frame after looping video file '{self.source}'.")
                            return False, None
                        return True, frame
                    elif self.is_file_source and not self.loop_video:
                        logger.info(f"Video file '{self.source}' ended. No loop.")
                        return False, None
                    else:
                        return False, None
                return True, frame
            except Exception as e:
                logger.error(f"Error reading frame: {str(e)}", exc_info=True)
                return False, None

    def _release_capture_internal(self):
        """Internal method to release capture, does not acquire lock."""
        if self.capture:
            logger.info(f"Releasing camera capture object for source: {getattr(self.capture, 'getSourceName', lambda: self.source)()}")
            self.capture.release()
        self.capture = None
        self._is_initialized = False

    def release_camera(self) -> int:
        """Public method to explicitly release the camera."""
        with self._operation_lock:
            logger.info(f"Explicitly releasing camera (source: {self.source}).")
            self._release_capture_internal()
            self._initialized_params = None
            logger.info("Camera released.")
        return SUCCESS

    def is_running(self) -> bool:
        with self._operation_lock:
            return self._is_initialized and self.capture is not None and self.capture.isOpened()

    def get_properties(self) -> dict:
        with self._operation_lock:
            if self.capture and self.capture.isOpened():
                return {
                    "source": self.source,
                    "width": self.capture.get(cv2.CAP_PROP_FRAME_WIDTH),
                    "height": self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT),
                    "fps": self.capture.get(cv2.CAP_PROP_FPS),
                    "is_file_source": self.is_file_source,
                    "loop_video": self.loop_video,
                    "is_opened": True,
                    "backend_name": self.capture.getBackendName() if hasattr(self.capture, 'getBackendName') else 'N/A'
                }
        return {
            "source": self.source,
            "width": self.width,
            "height": self.height,
            "fps": None,
            "is_file_source": self.is_file_source,
            "loop_video": self.loop_video,
            "is_opened": False,
            "backend_name": 'N/A'
        }

    def set_loop_video(self, loop: bool):
        with self._operation_lock:
            if self.loop_video != loop:
                self.loop_video = loop
                logger.info(f"Video loop behavior set to: {self.loop_video} for source {self.source}")
                if self._is_initialized and self._initialized_params:
                    self._initialized_params = (self._initialized_params[0], self._initialized_params[1], self._initialized_params[2], self.loop_video)


def get_camera_manager():
    return CameraManager()
