#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#_author = 'Yidian Lin'

"""
Module Description:
    用于实现车载多模态智能交互系统的个性化功能
"""

from applications.application import Application
from logger import Logger

# 模态包括:UI,语音,头部姿态,手势,视线
# 给视线实现一个特定场景,作为创新功能/展示用
# 头部姿态单纯承担场景中的确认和否定功能吧(返回0/1/None)
# UI初步不打算实现个性化(简化考虑)
# 语音模态和手势模态是个性化的关键
# 语音模态:可以给出相当复杂的字符串,在这里处理为功能(个性化的主要表现,将字符串转化为功能,例如播放音乐/Music代表Application.type.music)
# 和参数
# 手势模态:特定手势给出单一功能(可以个性化,包括个性化默认参数,类似快捷键的功能)

class Individuation:
    # TODO(): 个性化功能应该可以被设置(提供函数修改各个individuation_dict)
    # TODO(): 个性化功能应该通过写入文件实现持久化
    logger = Logger()
    speech_individuation_dict = {
        "播放音乐": Application.type.music_play,
        "导航": Application.type.navigation,
        "车辆状态": Application.type.vehicle_state
    }

    # TODO(): 测试来看,使用点头和摇头来控制功能实在太奇怪了,让头部模态单纯承担场景中的确认和否定功能吧(返回0/1/None)
    """
    head_individuation_dict = {
        "点头": Application.type.vehicle_state,
        "摇头": Application.type.navigation,
    }
    """
    # TODO(): 手势模态
    gesture_individuation_dict = {
        "0": Application.type.music_change_pause,
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
    # 手势个性化函数
    def gesture_individuation(cls, gesture_text: str) -> None:
        if gesture_text in cls.gesture_individuation_dict:
            function_name = cls.gesture_individuation_dict[gesture_text]
            Application.schedule(function_name,[])
            cls.logger.Log(f"模态:手势 功能:{Application.to_string(function_name)} 参数:{gesture_text}")
        else:
            print(f"无效语音命令: {gesture_text}")
            cls.logger.Log(f"模态:手势 功能:无效功能 参数:{gesture_text}")

    #@classmethod
    # 头部姿态个性化函数
    #def head_individuation(cls, headpos_text: str) -> None:
    #    # 修改以下代码,从speech_individuation_dict中获取对应的功能
    #    if headpos_text in cls.head_individuation_dict:
    #        function_name = cls.head_individuation_dict[headpos_text]
    #        Application.schedule(function_name,[])
    #        cls.logger.Log(f"模态:头部姿态 功能:{Application.to_string(function_name)} 参数:{headpos_text}")
    #    else:
    #        print(f"无效头部命令: {headpos_text}")
    #        cls.logger.Log(f"模态:头部姿态 功能:无效功能 参数:{headpos_text}")