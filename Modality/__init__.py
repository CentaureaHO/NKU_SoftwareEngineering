"""多模态交互系统模块，整合视觉、语音等多种交互方式"""

from .core import BaseModality, ModalityFactory, ModalityManager, ModalityState
from .speech import SpeechRecognition, SpeechState
from .visual import (BaseVisualModality, GestureState, GestureTracker,
                     HeadPoseState, HeadPoseTrackerGRU, VisualState)

__all__ = [
    # core
    'BaseModality', 'ModalityState', 'ModalityManager', 'ModalityFactory',

    # visual modality
    'BaseVisualModality', 'VisualState',
    'HeadPoseState', 'HeadPoseTrackerGRU',
    'GestureState', 'GestureTracker',

    # speech modality
    'SpeechRecognition', 'SpeechState',
]
