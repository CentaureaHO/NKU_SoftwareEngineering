"""头部姿态检测的常量和状态类"""
import math
import os
import sys
from typing import Any, Dict
from dataclasses import dataclass

import numpy as np

from modality.visual.base_visual import VisualState

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class HeadPoseParams:
    """头部姿态检测的参数常量"""
    VIDEO_FPS = 30  # 视频帧率
    HISTORY_LEN = 15  # 历史窗口大小，与GRU模型的window_size相同

    # 检测阈值
    NOD_THRESHOLD = 5.0          # 点头检测角度阈值 (度)
    SHAKE_THRESHOLD = 10.0       # 摇头检测角度阈值 (度)
    NOD_RATIO_THRESHOLD = 0.4    # 点头动作占比触发阈值
    SHAKE_RATIO_THRESHOLD = 0.4  # 摇头动作占比触发阈值
    NC_CHANGE_THRESHOLD = 0.025  # 鼻子-下巴距离变化阈值
    STABLE_THRESHOLD = 1.5       # 静止状态角度阈值

    # 当前状态
    STATUS_STATIONARY = "stationary"
    STATUS_NODDING = "nodding"
    STATUS_SHAKING = "shaking"
    STATUS_OTHER = "other"

    # 状态更新间隔
    STATUS_UPDATE_INTERVAL = 0.2  # 秒

    # 状态置信度阈值
    CONFIDENCE_THRESHOLD = 0.7

    # 最小检测所需历史数据量
    MIN_HISTORY_DATA = 3

    # MediaPipe 人脸关键点索引
    LANDMARK_NOSE = 1          # 鼻尖
    LANDMARK_CHIN = 152        # 下巴
    LANDMARK_LEFT_EYE = 159    # 左眼
    LANDMARK_RIGHT_EYE = 386   # 右眼
    LANDMARK_LEFT_EAR = 234    # 左耳
    LANDMARK_RIGHT_EAR = 454   # 右耳
    LANDMARK_LEFT_FACE = 206   # 左脸中心
    LANDMARK_RIGHT_FACE = 426  # 右脸中心

    # 面部关键点列表
    ESSENTIAL_LANDMARKS = [LANDMARK_NOSE, 33, 133, LANDMARK_LEFT_EYE, 145,
                           263, 362, 374, LANDMARK_RIGHT_EYE, 473, 468]


@dataclass
class HeadPoseState(VisualState):
    """头部姿态状态类，扩展自VisualState"""

    def __init__(self, frame=None, timestamp=None):
        super().__init__(frame, timestamp)
        self.detections = {
            "head_pose": {
                "pitch": 0.0,       # 俯仰角（点头）
                "yaw": 0.0,         # 偏航角（左右转头）
                "roll": 0.0,        # 翻滚角（头部倾斜）
                "detected": False,  # 是否检测到人脸
                "landmarks": []     # 关键点列表
            },
            "head_movement": {
                "is_nodding": False,        # 是否正在点头
                "is_shaking": False,        # 是否正在摇头
                "nod_confidence": 0.0,      # 点头动作的置信度
                "shake_confidence": 0.0,    # 摇头动作的置信度
                "status": HeadPoseParams.STATUS_STATIONARY,  # 当前状态
                "status_confidence": 0.0    # 当前状态的置信度
            }
        }

    def to_dict(self):
        result = super().to_dict()
        return result


def euclidean_dist(point1, point2):
    """计算二维点之间的欧几里得距离"""
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def rotation_matrix_to_angles(rotation_matrix):
    """
    从旋转矩阵计算欧拉角
    :param rotation_matrix: 3*3 旋转矩阵
    :return: 每个轴的角度(x, y, z) - 对应(pitch, yaw, roll)
    """
    x = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
    y = math.atan2(-rotation_matrix[2, 0], math.sqrt(rotation_matrix[0, 0] ** 2 +
                                                     rotation_matrix[1, 0] ** 2))
    z = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    return np.array([x, y, z]) * 180. / math.pi
