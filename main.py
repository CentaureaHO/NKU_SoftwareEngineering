#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# _author = 'Yidian Lin'

"""
Module Description:
    程序入口
"""

from system_init import initialize_system

# 初始化系统，获取所有组件
components = initialize_system()

if __name__ == "__main__":
    controller = components.get('controller')
    application = components.get('application')
    application.schedule(application.type.enter, [controller])
    controller.control()
