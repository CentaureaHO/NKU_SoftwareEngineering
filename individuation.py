#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# _author = 'Yidian Lin'

"""
Module Description:
    用于实现车载多模态智能交互系统的个性化功能
    支持的模态包括:物理模态(UI界面),语音模态,图像模态(头部姿态,手势,视线)
    主要支持语音模态和手势模态的个性化
    语音模态:可以给出相当复杂的字符串,映射为功能(将字符串转化为功能,例如"播放音乐"代表Application.type.music)
    手势模态:特定手势映射特定功能(可以个性化,包括个性化默认参数,类似快捷键的功能)
"""

import ast
import os
from typing import List

from applications.application import application


class Individuation:
    """个性化配置类"""

    def __init__(self) -> None:
        """构造函数"""
        # 获取记录个性化配置的文件路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        relative_path = "./database/individuation"
        dir_path = os.path.join(current_dir, relative_path)
        self.speech_individuation_file_path = os.path.join(
            dir_path, "speech_individuation.txt"
        )
        self.gesture_individuation_file_path = os.path.join(
            dir_path, "gesture_individuation.txt"
        )
        # 初始化手势个性化字典
        self.gesture_individuation_dict = {}
        if os.path.exists(self.gesture_individuation_file_path):
            with open(
                self.gesture_individuation_file_path, "r", encoding="utf-8"
            ) as file:
                self.gesture_individuation_dict = ast.literal_eval(file.read())
                for (
                    gesture_name,
                    application_type,
                ) in self.gesture_individuation_dict.items():
                    self.gesture_individuation_dict[gesture_name] = (
                        application.name2type[application_type]
                    )
        else:
            self.gesture_individuation_dict = {
                "握拳": application.type.music_change_pause,
                "挥手": application.type.music_change_pause,
                "竖起大拇指": application.type.music_change_pause,
            }
        print(f"gesture_individuation_dict: {self.gesture_individuation_dict}")
        # 初始化语音个性化字典
        self.speech_individuation_dict = {
            "播放音乐": application.type.music_play,
            "切换播放状态": application.type.music_change_pause,
            "导航": application.type.navigation,
            "车辆状态监测": application.type.monitor_jump,
        }
        if os.path.exists(self.speech_individuation_file_path):
            with open(
                self.speech_individuation_file_path, "r", encoding="utf-8"
            ) as file:
                speech_individuation_dict = ast.literal_eval(file.read())
                for application_type, speech_text in speech_individuation_dict.items():
                    if speech_text == "":
                        continue
                    self.speech_individuation_dict[speech_text] = application.name2type[
                        application_type
                    ]
        print(f"speech_individuation_dict: {self.speech_individuation_dict}")

    def speech_individuation(self, speech_text: str) -> None:
        """根据语音文本执行个性化功能"""
        if speech_text in self.speech_individuation_dict:
            function_name = self.speech_individuation_dict[speech_text]
            application.schedule(function_name, [])
        else:
            print(f"无效语音命令: {speech_text}")

    def gesture_individuation(self, gesture_text: str) -> None:
        """根据手势动作执行个性化功能"""
        if gesture_text in self.gesture_individuation_dict:
            function_name = self.gesture_individuation_dict[gesture_text]
            application.schedule(function_name, [])
        else:
            print(f"无效手势命令: {gesture_text}")

    def get_gesture_names(self) -> dict:
        """
        返回手势及其可选功能列表的字典，当前选定功能置于列表首位

        Returns:
            dict: 键为手势名称，值为功能列表(当前选定功能为列表第一项)
        """
        # 获取所有可用的应用功能名称
        all_applications = [
            application.to_string(app_type) for app_type in application.user_application
        ]

        # 创建结果字典
        gesture_config = {}

        # 为每个手势创建功能列表，将当前选定功能放在首位
        for gesture_name, app_type in self.gesture_individuation_dict.items():
            current_app = application.to_string(app_type)

            # 创建功能列表：当前功能 + 其他功能
            app_list = [current_app]
            for app in all_applications:
                if app != current_app:
                    app_list.append(app)

            # 添加到结果字典
            gesture_config[gesture_name] = app_list

        return gesture_config

    def get_speech_individuation_dict(self) -> List[str]:
        """获取语音个性化字典"""
        speech_individuation_dict = {}
        for speech_text, application_type in self.speech_individuation_dict.items():
            speech_individuation_dict[application.to_string(application_type)] = (
                speech_text
            )
        print(f"speech_individuation_dict: {speech_individuation_dict}")
        return speech_individuation_dict

    def set_gesture_individuation(self, gesture_options) -> None:
        """设置手势个性化字典"""
        self.gesture_individuation_dict = gesture_options
        print(f"self.gesture_individuation_dict: {
              self.gesture_individuation_dict}")
        with open(self.gesture_individuation_file_path, "w", encoding="utf-8") as f:
            f.write(str(self.gesture_individuation_dict))
        for gesture_name, application_type in gesture_options.items():
            self.gesture_individuation_dict[gesture_name] = application.name2type[
                application_type
            ]
        # print(f"gesture_individuation_dict: {gesture_individuation_dict}")
        # print(f"Individuation.dir_path: {Individuation.dir_path}")

    def set_speech_individuation(self, speech_options) -> None:
        """设置语音个性化字典"""
        print(f"speech_options: {speech_options}")
        for application_type, speech_text in speech_options.items():
            if speech_text != "":
                self.speech_individuation_dict[speech_text] = application.name2type[
                    application_type
                ]
        speech_options = {}
        for speech_text, application_type in self.speech_individuation_dict.items():
            speech_options[application.type2name[application_type]
                           ] = speech_text
        with open(self.speech_individuation_file_path, "w", encoding="utf-8") as f:
            f.write(str(speech_options))


individuation = Individuation()
