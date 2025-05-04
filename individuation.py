#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#_author = 'Yidian Lin'

"""
Module Description:
    用于实现车载多模态智能交互系统的个性化功能
"""

import os
import threading
from typing import List
from enum import Enum
from multimodal_controller import MultimodalController
from applications.application import Application
from logger import Logger

class Individuation:
    # TODO(): 个性化功能应该可以被设置(提供函数修改各个individuation_dict)
    # TODO(): 个性化功能应该通过写入文件实现持久化
    logger = Logger()
    speech_individuation_dict = {
        "播放音乐": Application.type.music,
        "导航": Application.type.navigation,
        "车辆状态": Application.type.vehicle_state
    }

    head_individuation_dict = {
        "点头": Application.type.vehicle_state,
        "摇头": Application.type.navigation,
    }

    def __init__(self) -> None:
        pass

    @classmethod
    # 语音个性化函数
    def speech_individuation(cls, speech_text: str) -> None:
        if speech_text in cls.speech_individuation_dict:
            function_name = cls.speech_individuation_dict[speech_text]
            Application.schedule(function_name,[])
            cls.logger.Log(f"模态:语音 功能:{Application.to_string(function_name)} 参数:{speech_text}")
        else:
            print(f"无效语音命令: {speech_text}")
            cls.logger.Log(f"模态:语音 功能:无效功能 参数:{speech_text}")

    @classmethod
    # 头部姿态个性化函数
    def head_individuation(cls, headpos_text: str) -> None:
        # 修改以下代码,从speech_individuation_dict中获取对应的功能
        if headpos_text in cls.head_individuation_dict:
            function_name = cls.head_individuation_dict[headpos_text]
            Application.schedule(function_name,[])
            cls.logger.Log(f"模态:头部姿态 功能:{Application.to_string(function_name)} 参数:{headpos_text}")
        else:
            print(f"无效头部命令: {headpos_text}")
            cls.logger.Log(f"模态:头部姿态 功能:无效功能 参数:{headpos_text}")