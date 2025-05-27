#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from system_init import initialize_system

# 初始化系统，获取所有组件
components = initialize_system()

# 在主模块中调用
if __name__ == "__main__":
    controller = components.get('controller')
    application = components.get('application')
    #time.sleep(10)
    #from viewer.viewer import navigate, jump_to_page
    #navigate()
    #jump_to_page("navigation")
    application.schedule(application.type.enter, [controller])
    while True:
        controller.control()
        #application.schedule(application.type.enter, [controller])
        #time.sleep(10)
    #controller.control()