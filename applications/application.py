#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#_author = 'Yidian Lin'

"""
Module Description:
    用于实现车载多模态智能交互系统的应用功能
"""

import os
from typing import List
from enum import Enum
from datetime import datetime
from .music import Music
from .navigation import Navigation
from .vehicle_state import VehicleState
import requests
from .abnormal import Abnormal

class Application:
    music = Music()
    type = Enum("type", ["music_getlist","music_play","music_pause","music_unpause","music_change_pause",
                         "navigation",
                         "vehicle_state",
                         "abnormal_distraction_reminder",
                        ])
    user_application = [type.music_play,type.music_change_pause,
                        type.navigation,
                        type.vehicle_state]
    
    def __init__(self) -> None:
        pass    

    @classmethod
    def get_application_name(cls) -> List[str]:
        application_names = []
        for type in cls.user_application:
            application_names.append(cls.to_string(type))
        return application_names
    
    # TODO(): 应该考虑不同功能之间的冲突问题
    @classmethod
    def schedule(cls,application_type: Enum,args: List) -> str:
        if application_type == Application.type.music_getlist:
            return cls.music.getlist()
        elif application_type == Application.type.music_play:
            requests.post('http://127.0.0.1:5000/trigger_action', json={'action': 'music'})
            if len(args) == 0:
                cls.music.play()
            else:
                cls.music.play(args[0])
        elif application_type == Application.type.music_pause:
            cls.music.pause()
        elif application_type == Application.type.music_unpause:
            cls.music.unpause()
        elif application_type == Application.type.music_change_pause:
            cls.music.change_pause()
        elif application_type == Application.type.navigation:
            navigation = Navigation()
            navigation.show("南开大学津南校区","南开大学八里台校区")
        elif application_type == Application.type.vehicle_state:
            state = VehicleState()
            print(state.monitor(VehicleState.type.oil_quantity))
        elif application_type == Application.type.abnormal_distraction_reminder:
            assert(len(args) == 1)
            Abnormal.distraction_reminder(args[0])
        return None

    @classmethod
    def to_string(cls,application_type: Enum) -> str:
        if application_type == Application.type.music_getlist:
            return "获取音乐列表"
        elif application_type == Application.type.music_play:
            return "播放音乐"
        elif application_type == Application.type.music_pause:
            return "暂停音乐"
        elif application_type == Application.type.music_unpause:
            return "继续播放音乐"
        elif application_type == Application.type.music_change_pause:
            return "切换播放状态"
        elif application_type == Application.type.navigation:
            return "导航"
        elif application_type == Application.type.vehicle_state:
            return "车辆状态"
        elif application_type == Application.type.abnormal_distraction_reminder:
            return "异常分心提醒"
        else:
            return "未知功能"


if __name__ == '__main__':
    app = Application()
    app.schedule(Application.type.music,[])