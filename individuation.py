#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#_author = 'Yidian Lin'

"""
Module Description:
    用于实现车载多模态智能交互系统的个性化功能
"""

from applications.application import Application
from logger import logger
from typing import List
import os
import pickle

# 支持的模态包括:物理模态(UI界面),语音模态,图像模态(头部姿态,手势,视线)
# 主要支持语音模态和手势模态的个性化
# 语音模态:可以给出相当复杂的字符串,映射为功能(将字符串转化为功能,例如播放音乐/Music代表Application.type.music)
# 手势模态:特定手势映射特定功能(可以个性化,包括个性化默认参数,类似快捷键的功能)

class Individuation:
    def __init__(self) -> None:
        # 个性化配置文件路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        relative_path = "./database/individuation"
        dir_path = os.path.join(current_dir, relative_path)
        self.speech_individuation_file_path = os.path.join(dir_path, 'speech_individuation.txt')
        self.gesture_individuation_file_path = os.path.join(dir_path, 'gesture_individuation.txt')
        # 手势个性化字典
        self.gesture_individuation_dict = {}
        if os.path.exists(self.gesture_individuation_file_path):
            with open(self.gesture_individuation_file_path, 'r' , encoding = "utf-8" ) as file:  
                self.gesture_individuation_dict = eval(file.read())
                for gesture_name, application_type in self.gesture_individuation_dict.items():
                    self.gesture_individuation_dict[gesture_name] = Application.name2type[application_type]
        else:
            self.gesture_individuation_dict = {
                "握拳": Application.type.music_change_pause,
                "挥手": Application.type.music_change_pause,
                "竖起大拇指":Application.type.music_change_pause
            }
        print(f"gesture_individuation_dict: {self.gesture_individuation_dict}")
        # 语音个性化字典
        self.speech_individuation_dict = {}
        if os.path.exists(self.speech_individuation_file_path):
            with open(self.speech_individuation_file_path, 'r' , encoding = "utf-8" ) as file:  
                speech_individuation_dict = eval(file.read())
                for application_type , speech_text in speech_individuation_dict.items():
                    if speech_text == "":
                        continue
                    self.speech_individuation_dict[speech_text] = Application.name2type[application_type]
        else:
            self.speech_individuation_dict = {
                "播放音乐" : Application.type.music_play,
                "切换播放状态" : Application.type.music_change_pause,
                "导航" : Application.type.navigation,
                "车辆状态监测" : Application.type.vehicle_state
            }
        print(f"speech_individuation_dict: {self.speech_individuation_dict}")

    # 语音个性化函数
    def speech_individuation(self, speech_text: str) -> None:
        if speech_text in self.speech_individuation_dict:
            function_name = self.speech_individuation_dict[speech_text]
            Application.schedule(function_name,[])
            # cls.logger.Log(f"模态:语音 功能:{Application.to_string(function_name)} 参数:{speech_text}")
        else:
            print(f"无效语音命令: {speech_text}")
            # cls.logger.Log(f"模态:语音 功能:无效功能 参数:{speech_text}")

    # 手势个性化函数
    def gesture_individuation(self, gesture_text: str) -> None:
        if gesture_text in self.gesture_individuation_dict:
            function_name = self.gesture_individuation_dict[gesture_text]
            Application.schedule(function_name,[])
            #self.logger.Log(f"模态:手势 功能:{Application.to_string(function_name)} 参数:{gesture_text}")
        else:
            print(f"无效语音命令: {gesture_text}")
            #self.logger.Log(f"模态:手势 功能:无效功能 参数:{gesture_text}")

    def get_gesture_names(self) -> List[str]:
        gesture_names = []
        for gesture_name in self.gesture_individuation_dict.keys():
            gesture_names.append(gesture_name)
        return gesture_names
    
    def set_gesture_individuation(self, gesture_options) -> None:
        self.gesture_individuation_dict = gesture_options
        print(f"self.gesture_individuation_dict: {self.gesture_individuation_dict}")
        # file_path = os.path.join(Individuation.dir_path, 'gesture_individuation.txt')
        with open(self.gesture_individuation_file_path, 'w' , encoding = "utf-8" ) as f:
            f.write(str(self.gesture_individuation_dict))
        for gesture_name, application_type in gesture_options.items():
            self.gesture_individuation_dict[gesture_name] = Application.name2type[application_type]
        # print(f"gesture_individuation_dict: {gesture_individuation_dict}")
        # print(f"Individuation.dir_path: {Individuation.dir_path}")

    def set_speech_individuation(self, speech_options) -> None:
        self.speech_individuation_dict = {}
        # self.speech_individuation_dict = speech_options
        print(f"speech_options: {speech_options}")
        with open(self.speech_individuation_file_path, 'w' , encoding = "utf-8" ) as f:
            f.write(str(speech_options))
        for application_type , speech_text in speech_options.items():
            self.speech_individuation_dict[speech_text] = Application.name2type[application_type]
        
        
individuation = Individuation()

if __name__ == '__main__':
    test_gesture_individuation_dict = {
        "握拳": '播放音乐',
        "挥手": '播放音乐',
        "竖起大拇指":'播放音乐'
    }
    individuation.set_gesture_individuation(test_gesture_individuation_dict)