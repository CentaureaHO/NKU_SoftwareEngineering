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

class Application:
    music = Music()
    type = Enum("type", ["music_getlist","music_play","music_pause","music_unpause","music_change_pause","navigation","vehicle_state"])
    
    def __init__(self) -> None:
        pass    
    
    # TODO(): 应该考虑不同功能之间的冲突问题
    @classmethod
    def schedule(cls,application_type: Enum,args: List) -> str:
        if application_type == Application.type.music_getlist:
            return cls.music.getlist()
        elif application_type == Application.type.music_play:
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
        return None

    @classmethod
    def to_string(cls,application_type: Enum) -> str:
        if application_type == Application.type.music:
            return "音乐"
        elif application_type == Application.type.navigation:
            return "导航"
        elif application_type == Application.type.vehicle_state:
            return "车辆状态"
        else:
            return "未知功能"

if __name__ == '__main__':
    app = Application()
    app.schedule(Application.type.music,[])
