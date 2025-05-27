#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# _author = 'Yidian Lin'

"""
Module Description:
    用于实现车载多模态智能交互系统的应用功能
"""


from typing import List
from enum import Enum
from logger import logger
from .music import Music
from .navigation import Navigation
from .vehicle_state import VehicleState
from .enter import Enter



class Application:
    """实现车载多模态智能交互系统的应用功能"""

    type = Enum(
        "type",
        [
            "music_getlist",
            "music_play",
            "music_pause",
            "music_unpause",
            "music_change_pause",
            "navigation_getlist",
            "navigation",
            "vehicle_state",
            "monitor_getlist",
            "monitor_jump",
            "abnormal_distraction_reminder",
            "enter",
        ],
    )
    user_application = [
        type.music_play,
        type.music_change_pause,
        type.navigation,
        type.monitor_jump,
    ]
    type2name = {
        type.music_getlist: "获取音乐列表",
        type.music_play: "播放音乐",
        type.music_pause: "暂停音乐",
        type.music_unpause: "继续播放音乐",
        type.music_change_pause: "切换播放状态",
        type.navigation_getlist: "获取导航列表",
        type.navigation: "导航",
        type.vehicle_state: "车辆状态监测(已废弃)",
        type.monitor_getlist: "获取监测列表",
        type.monitor_jump: "车辆状态监测",
        type.abnormal_distraction_reminder: "异常分心提醒",
        type.enter: "系统启动时自动调用应用功能",
    }
    name2type = {}
    for key, value in type2name.items():
        name2type[value] = key

    def __init__(self) -> None:
        """构造函数"""
        self.music = Music()
        self.enter = Enter()

    def get_application_names(self) -> List[str]:
        """获取用户应用功能名称"""
        application_names = []
        for type_ in self.user_application:
            application_names.append(self.to_string(type_))
        return application_names

    def schedule(self, application_type: Enum, args: List) -> str:
        """调度应用功能"""
        if application_type == Application.type.music_getlist:
            return self.music.getlist()
        if application_type == Application.type.music_play:
            logger.log("应用功能:播放音乐")
            if len(args) == 0:
                self.music.play()
            else:
                self.music.play(args[0])
        if application_type == Application.type.music_pause:
            self.music.pause()
        if application_type == Application.type.music_unpause:
            self.music.unpause()
        if application_type == Application.type.music_change_pause:
            logger.log("应用功能:切换音乐播放状态")
            self.music.change_pause()
        if application_type == Application.type.navigation:
            logger.log("应用功能:进行导航")
            navigation = Navigation()
            navigation.navigate()
        if application_type == Application.type.monitor_getlist:
            state = VehicleState()
            return state.monitor()
        if application_type == Application.type.monitor_jump:
            from viewer.viewer import jump_to_page

            jump_to_page("status")
            logger.log("应用功能:进行车辆状态监测")
        if application_type == Application.type.enter:
            assert len(args) == 1
            self.enter.enter(args[0])
        return None

    def to_string(self, application_type: Enum) -> str:
        """将应用功能枚举类型转换为字符串"""
        if application_type in self.type2name:
            return self.type2name[application_type]
        return "未知功能"


application = Application()
