#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#_author = 'Yidian Lin'

"""
Module Description:
    用于实现车载多模态智能交互系统的日志功能
"""

import os
from typing import List
from enum import Enum
from datetime import datetime

class Logger:
    # type = Enum("type", ["music","navigation","vehicle_state"])
    def __init__(self) -> None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        relative_path = "database/log/Log.txt"
        self.log_path = os.path.join(current_dir, relative_path)

    # TODO():暂定格式为 时间 模态 功能 参数 
    def Log(self,log: str) -> None:
        current_time = datetime.now()
        time_string = current_time.strftime("[%Y-%m-%d %H:%M:%S]")
        print(log)
        with open(self.log_path, 'a+' ,encoding='utf-8') as file:
            file.write(time_string)
            file.write(log)
            file.write("\n")