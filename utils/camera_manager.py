#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# _author = 'Xianda Tang'

"""
Module Description:
    统一管理前置摄像头资源
"""

import logging
import os
import queue
import threading
import time
from typing import Optional, Tuple, Union, Any # Added Any
from dataclasses import dataclass, field # Added dataclasses

import cv2
import numpy as np

try:
    from Modality.core.error_codes import (CAMERA_NOT_AVAILABLE, SUCCESS,
                                           VIDEO_FILE_NOT_FOUND,
                                           VIDEO_SOURCE_ERROR)
except ImportError:
    logging.warning(
        "Failed to import error codes from Modality.core. Using fallback definitions."
    )
    SUCCESS = 0
    CAMERA_NOT_AVAILABLE = 101
    VIDEO_FILE_NOT_FOUND = 102
    VIDEO_SOURCE_ERROR = 103
    FRAME_ACQUISITION_FAILED = 104

logger = logging.getLogger("CameraManager")


@dataclass
class CameraConfig:
    """
    Holds configuration for camera capture, including source, width, height,
    """
    source: Union[int, str] = 0
    width: int = 640
    height: int = 480
    loop_video: bool = True
    is_file_source: bool = field(init=False)

    def __post_init__(self):
        """Initializes 'is_file_source' based on the type of 'source'."""
        self.is_file_source = isinstance(self.source, str)

    def to_tuple(self) -> Tuple[Union[int, str], int, int, bool]:
        """Returns the configuration as a tuple."""
        return (self.source, self.width, self.height, self.loop_video)


@dataclass
class CaptureState:
    """
    Holds the state of the camera capture, including buffer, thread, and frames.
    """
    capture: Optional[cv2.VideoCapture] = None
    frame_buffer: queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=10))
    latest_frame: Optional[np.ndarray] = None
    capture_thread: Optional[threading.Thread] = None
    stop_capture_event: threading.Event = field(default_factory=threading.Event)
    is_initialized: bool = False
    initialized_config_tuple: Optional[Tuple[Union[int, str], int, int, bool]] = None


class CameraManager:
    """单例摄像头管理器类，用于统一管理前置摄像头资源"""
    _instance = None
    _class_lock = threading.Lock()

    def __new__(cls, *args: Any, **kwargs: Any) -> 'CameraManager':
        if not cls._instance:
            with cls._class_lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        # Ensure _initialized_once is defined before first access
        if not hasattr(self, "_initialized_once"):
            self._initialized_once = False

        if self._initialized_once:
            return

        self.config = CameraConfig() # Default config
        self.capture_state = CaptureState()
        self._operation_lock = threading.Lock()
        self._initialized_once = True
        logger.info("CameraManager instance created/accessed.")

    def _open_and_configure_capture(self) -> bool:
        """Opens the capture source and configures it. Returns True on success."""
        if self.config.is_file_source:
            if not os.path.isfile(self.config.source):
                logger.error("Video file not found: %s", self.config.source)
                return False
            self.capture_state.capture = cv2.VideoCapture(self.config.source)
        else:
            self.capture_state.capture = cv2.VideoCapture(self.config.source)

        if not self.capture_state.capture or not self.capture_state.capture.isOpened():
            logger.error("Failed to open video source: %s", self.config.source)
            self.capture_state.capture = None
            return False

        self.capture_state.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        self.capture_state.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        return True

    def _handle_frame_read_failure(self, retry_count: int, max_retries: int) -> Tuple[bool, int]:
        """Handles frame read failure, including looping and retries. 
        Returns (should_continue, new_retry_count)"""
        if self.config.is_file_source and self.config.loop_video:
            logger.info("Video file '%s' ended. Looping...", self.config.source)
            if self.capture_state.capture:
                self.capture_state.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return True, 0  # Continue looping, reset retry_count

        logger.warning("Failed to read frame. Retrying...")
        retry_count += 1
        if retry_count > max_retries:
            logger.error("Failed to read frame after %s attempts", max_retries)
            return False, retry_count # Stop retrying
        time.sleep(1.0) # retry_delay
        return True, retry_count # Continue retrying

    def _capture_frames(self) -> None:
        """持续捕获帧并放入缓冲区的线程函数"""
        logger.info("Frame capture thread started for source: %s", self.config.source)
        retry_count = 0
        max_retries = 5
        retry_delay = 1.0

        while not self.capture_state.stop_capture_event.is_set():
            if not self.capture_state.capture or not self.capture_state.capture.isOpened():
                logger.warning("Camera not opened. Attempting to reconnect...")
                with self._operation_lock:
                    if not self._open_and_configure_capture():
                        retry_count += 1
                        if retry_count > max_retries:
                            logger.error("Failed to reconnect after %s attempts", max_retries)
                            break
                        time.sleep(retry_delay)
                        continue
                    retry_count = 0  # Reset on successful reconnect
                    logger.info("Successfully reconnected to camera")

            try:
                ret, frame = self.capture_state.capture.read()
                if not ret:
                    should_continue, retry_count = self._handle_frame_read_failure(
                        retry_count, max_retries)
                    if not should_continue:
                        break
                    continue

                retry_count = 0 # Reset on successful read

                with self._operation_lock:
                    self.capture_state.latest_frame = frame.copy()

                try:
                    if self.capture_state.frame_buffer.full():
                        try:
                            self.capture_state.frame_buffer.get_nowait()
                        except queue.Empty:
                            pass
                    self.capture_state.frame_buffer.put_nowait(frame)
                except BaseException:  # pylint: disable=broad-exception-caught
                    pass
                time.sleep(0.01)

            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.error("Error in capture thread: %s", str(e), exc_info=True)
                retry_count += 1
                if retry_count > max_retries:
                    logger.error("Too many errors in capture thread. Exiting.")
                    break
                time.sleep(retry_delay)

        with self._operation_lock:
            if self.capture_state.capture:
                self.capture_state.capture.release()
                self.capture_state.capture = None
        logger.info("Frame capture thread stopped")

    def initialize_camera(
        self,
        source: Union[int, str] = 0,
        width: int = 640,
        height: int = 480,
        loop_video: bool = True,
    ) -> int:
        """
        Initializes or re-initializes the camera with the specified parameters.

        If the camera is already initialized with the same parameters and running,
        this method will return SUCCESS without re-initializing. Otherwise, it
        stops any existing capture, configures the camera with the new parameters,
        and starts a new capture thread.

        Args:
            source: Camera index (int) or video file path (str).
            width: Desired frame width.
            height: Desired frame height.
            loop_video: If True and source is a file, the video will loop.

        Returns:
            An integer error code (e.g., SUCCESS, CAMERA_NOT_AVAILABLE).
        """
        with self._operation_lock:
            new_config = CameraConfig(source, width, height, loop_video)
            current_config_tuple = new_config.to_tuple()

            if (
                self.capture_state.is_initialized
                and self.capture_state.initialized_config_tuple == current_config_tuple
                and self.capture_state.capture_thread
                and self.capture_state.capture_thread.is_alive()
            ):
                logger.info(
                    "Camera already initialized and running with same parameters: %s",
                    current_config_tuple,
                )
                return SUCCESS

            self._stop_existing_thread() # Resets capture_state.is_initialized

            self.config = new_config # Update manager's config

            logger.info(
                "Initializing camera: source=%s, width=%s, height=%s, is_file=%s, loop=%s",
                self.config.source, self.config.width, self.config.height,
                self.config.is_file_source, self.config.loop_video
            )

            try:
                if not self._open_and_configure_capture():
                    self.capture_state.is_initialized = False
                    return (
                        VIDEO_FILE_NOT_FOUND if self.config.is_file_source
                        else CAMERA_NOT_AVAILABLE
                    )

                actual_width = self.capture_state.capture.get(cv2.CAP_PROP_FRAME_WIDTH)
                actual_height = self.capture_state.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
                logger.info(
                    "Camera opened. Requested: %sx%s, Actual: %sx%s",
                    self.config.width, self.config.height,
                    int(actual_width), int(actual_height)
                )

                # Clear buffer before starting new thread
                while not self.capture_state.frame_buffer.empty():
                    try:
                        self.capture_state.frame_buffer.get_nowait()
                    except BaseException:  # pylint: disable=broad-exception-caught
                        pass

                self.capture_state.stop_capture_event.clear()
                self.capture_state.capture_thread = threading.Thread(
                    target=self._capture_frames, daemon=True
                )
                self.capture_state.capture_thread.start()

                self.capture_state.is_initialized = True
                self.capture_state.initialized_config_tuple = current_config_tuple
                logger.info("Camera initialized successfully.")
                return SUCCESS

            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.error("Error initializing camera: %s", str(e), exc_info=True)
                self._release_capture_internal() # Ensure cleanup
                return VIDEO_SOURCE_ERROR

    def _stop_existing_thread(self) -> None:
        """停止现有的捕获线程"""
        if self.capture_state.capture_thread and self.capture_state.capture_thread.is_alive():
            logger.info("Stopping existing capture thread...")
            self.capture_state.stop_capture_event.set()
            self.capture_state.capture_thread.join(timeout=2.0)
            if self.capture_state.capture_thread.is_alive():
                logger.warning("Capture thread did not stop in time - continuing anyway")

        if self.capture_state.capture:
            try:
                self.capture_state.capture.release()
            except BaseException:  # pylint: disable=broad-exception-caught
                pass
            self.capture_state.capture = None

        self.capture_state.capture_thread = None
        self.capture_state.is_initialized = False # Mark as not initialized

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """读取一帧。首先尝试从缓冲区获取，如果没有可用帧则返回最新帧或失败"""
        if not self.capture_state.is_initialized:
            return False, None

        try:
            frame = self.capture_state.frame_buffer.get_nowait()
            return True, frame
        except queue.Empty:
            pass

        with self._operation_lock:
            if self.capture_state.latest_frame is not None:
                return True, self.capture_state.latest_frame.copy()

        return False, None

    def _release_capture_internal(self) -> None:
        """内部方法：释放摄像头资源"""
        self._stop_existing_thread()
        self.capture_state.latest_frame = None
        while not self.capture_state.frame_buffer.empty():
            try:
                self.capture_state.frame_buffer.get_nowait()
            except BaseException:  # pylint: disable=broad-exception-caught
                pass
        # Reset initialized config tracking
        self.capture_state.initialized_config_tuple = None


    def release_camera(self) -> int:
        """释放摄像头资源"""
        with self._operation_lock:
            logger.info("Explicitly releasing camera (source: %s).", self.config.source)
            self._release_capture_internal()
            logger.info("Camera released.")
        return SUCCESS

    def is_running(self) -> bool:
        """检查摄像头是否在运行"""
        return (
            self.capture_state.is_initialized
            and self.capture_state.capture_thread is not None
            and self.capture_state.capture_thread.is_alive()
        )

    def get_properties(self) -> dict:
        """获取摄像头属性"""
        with self._operation_lock:
            if self.capture_state.capture and self.is_running():
                return {
                    "source": self.config.source,
                    "width": self.capture_state.capture.get(cv2.CAP_PROP_FRAME_WIDTH),
                    "height": self.capture_state.capture.get(cv2.CAP_PROP_FRAME_HEIGHT),
                    "fps": self.capture_state.capture.get(cv2.CAP_PROP_FPS),
                    "is_file_source": self.config.is_file_source,
                    "loop_video": self.config.loop_video,
                    "is_opened": True,
                    "backend_name": (
                        self.capture_state.capture.getBackendName()
                        if hasattr(self.capture_state.capture, "getBackendName")
                        else "N/A"
                    ),
                    "buffer_size": self.capture_state.frame_buffer.qsize(),
                }
        return {
            "source": self.config.source,
            "width": self.config.width,
            "height": self.config.height,
            "fps": None,
            "is_file_source": self.config.is_file_source,
            "loop_video": self.config.loop_video,
            "is_opened": False,
            "backend_name": "N/A",
            "buffer_size": 0,
        }

    def set_loop_video(self, loop: bool) -> None:
        """设置视频循环播放"""
        with self._operation_lock:
            if self.config.loop_video != loop:
                self.config.loop_video = loop
                logger.info(
                    "Video loop behavior set to: %s for source %s",
                    self.config.loop_video, self.config.source
                )
                if self.capture_state.is_initialized \
                    and self.capture_state.initialized_config_tuple:
                    # Update the stored initialized_params tuple
                    src, w, h, _ = self.capture_state.initialized_config_tuple
                    self.capture_state.initialized_config_tuple = (
                        src, w, h, self.config.loop_video)


def get_camera_manager() -> CameraManager:
    """获取 CameraManager 的单例实例。"""
    return CameraManager()
