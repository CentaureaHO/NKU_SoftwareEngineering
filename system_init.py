#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# _author = 'Yidian Lin'

"""
Module Description:
    用于存储所有系统组件
"""

import threading
import time
import webbrowser

from logger import logger
from utils.tools import speecher_player

# 解决 C0415: Import outside toplevel
from applications.application import application
from multimodal_controller import MultimodalController
from setting import Setting
from individuation import individuation
from viewer.viewer import start_flask_server

# 全局组件字典，存储所有系统组件
_components = {}


def get_component(name):
    """获取已初始化的系统组件"""
    return _components.get(name)


def register_component(name, component):
    """注册系统组件"""
    _components[name] = component
    return component


def initialize_system():
    """系统初始化入口函数"""
    print("=== 正在初始化车载多模态智能交互系统 ===")
    logger.log("系统初始化")
    speecher_player.speech_synthesize_sync("欢迎使用车载多模态智能交互系统")
    speecher_player.speech_synthesize_sync("正在初始化系统,请耐心等待...")

    # 1. 初始化应用程序控制器
    register_component('application', application)
    print("✓ 应用程序控制器初始化完成")

    # 2. 初始化多模态控制器
    controller = MultimodalController()
    register_component('controller', controller)
    print("✓ 多模态控制器初始化完成")

    # 3. 初始化设置模块
    setting = Setting(controller.speecher)
    register_component('setting', setting)
    print("✓ 设置模块初始化完成")

    # 4. 初始化个性化配置
    register_component('individuation', individuation)
    print("✓ 个性化配置初始化完成")

    # 5. 启动Web界面
    flask_thread = threading.Thread(target=start_flask_server)
    flask_thread.daemon = True
    flask_thread.start()
    print("✓ Web服务器启动完成")

    # 等待Flask服务器启动
    time.sleep(5)

    # 6. 打开浏览器
    print("✓ 正在打开浏览器...")
    webbrowser.open("http://127.0.0.1:5000")
    time.sleep(3)
    speecher_player.speech_synthesize_sync("物理模态(UI界面)初始化完成")
    logger.log("物理模态(UI界面)初始化完成")

    print("=== 系统初始化完成 ===")
    speecher_player.speech_synthesize_sync("系统初始化完成")
    return _components
