"""模态核心包，提供模态管理、创建和基础定义。"""
from .base_modality import BaseModality, ModalityState
from .error_codes import *
from .modality_factory import ModalityFactory
from .modality_manager import ModalityManager

__all__ = [
    'BaseModality', 'ModalityState',
    'ModalityManager', 'ModalityFactory',

    # 错误码
    'SUCCESS', 'UNKNOWN_ERROR', 'INVALID_ARGUMENT', 'NOT_INITIALIZED',
    'ALREADY_INITIALIZED', 'OPERATION_FAILED', 'NOT_IMPLEMENTED',
    'RESOURCE_UNAVAILABLE', 'TIMEOUT', 'RUNTIME_ERROR', 'PERMISSION_DENIED',
    'MODALITY_NOT_FOUND', 'MODALITY_ALREADY_EXISTS', 'MODALITY_REGISTRATION_FAILED',
    'MODALITY_START_FAILED', 'MODALITY_STOP_FAILED', 'VIDEO_SOURCE_ERROR',
    'CAMERA_NOT_AVAILABLE', 'VIDEO_FILE_NOT_FOUND', 'FRAME_ACQUISITION_FAILED',
    'VIDEO_PROCESSING_ERROR', 'FACE_DETECTION_ERROR', 'HEAD_TRACKING_FAILED',
    'MEDIAPIPE_INITIALIZATION_FAILED', 'get_error_message'
]
