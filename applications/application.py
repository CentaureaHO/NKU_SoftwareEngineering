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
from .music import Music
from .navigation import Navigation
from .vehicle_state import VehicleState
from .enter import Enter
import requests
#from .abnormal import Abnormal

class Application:
    type = Enum("type", ["music_getlist","music_play","music_pause","music_unpause","music_change_pause",
                         "navigation_getlist","navigation",
                         "vehicle_state",
                         "abnormal_distraction_reminder",
                         "enter",
                        ])
    user_application = [type.music_play,type.music_change_pause,
                        type.navigation,
                        type.vehicle_state]
    type2name = {
        type.music_getlist: "获取音乐列表",
        type.music_play: "播放音乐",
        type.music_pause: "暂停音乐",
        type.music_unpause: "继续播放音乐",
        type.music_change_pause: "切换播放状态",
        type.navigation_getlist: "获取导航列表",
        type.navigation: "导航",
        type.vehicle_state: "车辆状态监测",
        type.abnormal_distraction_reminder: "异常分心提醒",
        type.enter: "系统启动时自动调用应用功能",
    }
    name2type = {}
    for key, value in type2name.items():
        name2type[value] = key

    def __init__(self) -> None:
        self.music = Music()
        self.enter = Enter()

    def get_application_names(self) -> List[str]:
        application_names = []
        for type in self.user_application:
            application_names.append(self.to_string(type))
        return application_names

    def schedule(self,application_type: Enum,args: List) -> str:
        if application_type == Application.type.music_getlist:
            return self.music.getlist()
        elif application_type == Application.type.music_play:
            requests.post('http://127.0.0.1:5000/trigger_action', json={'action': 'music'})
            if len(args) == 0:
                self.music.play()
            else:
                self.music.play(args[0])
        elif application_type == Application.type.music_pause:
            self.music.pause()
        elif application_type == Application.type.music_unpause:
            self.music.unpause()
        elif application_type == Application.type.music_change_pause:
            self.music.change_pause()
        elif application_type == Application.type.navigation_getlist:
            navigation = Navigation()
            return navigation.getlist()
        elif application_type == Application.type.navigation:
            navigation = Navigation()
            #navigation.show("南开大学津南校区","南开大学八里台校区")
            return navigation.navigate(args[0])
        elif application_type == Application.type.vehicle_state:
            state = VehicleState()
            return state.monitor()
            #print(state.monitor(VehicleState.type.oil_quantity))
        #elif application_type == Application.type.abnormal_distraction_reminder:
            #assert(len(args) == 1)
            #Abnormal.distraction_reminder(args[0])
        elif application_type == Application.type.enter:
            assert(len(args) == 1)
            self.enter.enter(args[0])
        return None

    def to_string(self,application_type: Enum) -> str:
        if application_type in self.type2name:
            return self.type2name[application_type]
        else:
            return "未知功能"

application = Application()

if __name__ == '__main__':
    app = Application()
    app.schedule(Application.type.music,[])