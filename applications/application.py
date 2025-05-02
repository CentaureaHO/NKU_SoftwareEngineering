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
from music import Music
from navigation import Navigation
from vehicle_state import VehicleState

class Application:
    type = Enum("type", ["music","navigation","vehicle_state"])
    def __init__(self) -> None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        relative_path = "../database/log/Log.txt"
        self.log_path = os.path.join(current_dir, relative_path)

    def schedule(self,application_type: Enum,args: List) -> str:
        if application_type == Application.type.music:
            self.Log("播放音乐——南开校歌音频.mp3")
            music = Music()
            music.play("南开校歌")
        elif application_type == Application.type.navigation:
            navigation = Navigation()
            navigation.show("南开大学津南校区","南开大学八里台校区")
        elif application_type == Application.type.vehicle_state:
            state = VehicleState()
            print(state.monitor(VehicleState.type.oil_quantity))

    def Log(self,log: str) -> None:
        current_time = datetime.now()
        time_string = current_time.strftime("[%Y-%m-%d %H:%M:%S]")
        print(log)
        with open(self.log_path, 'a+' ,encoding='utf-8') as file:
            file.write(time_string)
            file.write(log)
            file.write("\n")

if __name__ == '__main__':
    app = Application()
    app.schedule(Application.type.music,[])