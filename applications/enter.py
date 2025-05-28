#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#_author = 'Yidian Lin'

"""
Module Description:
    实现启动系统时自动开启应用功能
"""

import os
import time
import requests
import sys
sys.path.append(r'C:\2025spring\软件工程\小组作业\NKU_SoftwareEngineering')
from utils.tools import speecher_player
from viewer.viewer import update_note, update_light
from logger import logger

class Enter:
    def __init__(self) -> None:
        pass

    def enter(self,controller) -> None:
        logger.Log("典型场景:启动系统时自动开启应用功能")
        from system_init import get_component
        application = get_component('application')
        setting = get_component('setting')
        text = "系统初始化完毕，请驾驶员目视前方"
        update_note(text)
        update_light("red", True)
        # 视线直视前方超过3秒
        tag = False
        start_time = None
        while True:
            state = controller.gazer.get_key_info()
            # print(f"获取到的状态: {state}")
            if state != "中间":
                tag = False
                start_time = None
                continue
            if tag == False:
                tag = True
                start_time = time.time()
            elif tag == True and time.time() - start_time > 3:
                break
        logger.Log("视觉模态(视线方向部分):视线直视前方超过3秒")
        
        text = "驾驶员已就位"
        update_note(text)
        update_light("green", False)
        # time.sleep(1)
        speecher_player.speech_synthesize_sync(text)
        func = [application.type.monitor_jump, application.type.navigation, application.type.music_play]
        speecher_str1 = ["监测车辆状态",
                         "默认导航:从南开大学津南校区到八里台校区",
                         "播放默认音乐:南开校歌",]
        speecher_str2 = ["监测","导航","播放音乐",]
        # func = [application.type.navigation]
        # speecher_str = ["进行导航"]

        for i in range(len(func)):
            time.sleep(1)
            text = f"是否为您{speecher_str1[i]}"
            text = f"是否为您{speecher_str1[i]}\n" + \
                f"同意请语音输入\"同意{speecher_str2[i]}\"、竖起大拇指或点头\n" \
                f"拒绝请语音输入\"拒绝{speecher_str2[i]}\"、摇手或摇头"
            update_note(text)
            update_light("green", True)
            time.sleep(1)
            speecher_player.speech_synthesize_sync(text)

            tag = None
            while tag == None:
                # 获取语音输入
                is_driver = False
                state = controller.speecher.get_key_info()
                driver_state = controller.speecher.update()
                if driver_state and driver_state.recognition['speaker_name']:
                    print(f"说话人: {driver_state.recognition['speaker_name']}")
                    print(f"驾驶员: {setting.get_driver()}")
                    is_driver = (driver_state.recognition['speaker_name'] == setting.get_driver())

                if state != None:
                    print("语音输入:",state)
                    logger.Log(f"语音模态:输入\"{state}\"")
                if state == f"同意{speecher_str2[i]}" and is_driver == True:
                    tag = True
                    break
                elif state == f"拒绝{speecher_str2[i]}" and is_driver == True:
                    tag = False
                    break
                # 获取手势输入
                state = controller.static_gesture_tracker.get_key_info()
                if state == "竖起大拇指":
                    print("手势输入:竖起大拇指")
                    logger.Log(f"视觉模态(手势动作部分):竖起大拇指")
                    tag = True
                    break
                elif state == "摇手":
                    print("手势输入:摇手")
                    logger.Log(f"视觉模态(手势动作部分):摇手")
                    tag = False
                    break
                # 获取头部姿态输入
                state = controller.headposer.get_key_info()
                if state == "点头":
                    print("头部姿态输入:点头")
                    logger.Log(f"视觉模态(头部姿态部分):点头")
                    tag = True
                    break
                elif state == "摇头":
                    print("头部姿态输入:摇头")
                    logger.Log(f"视觉模态(头部姿态部分):摇头")
                    tag = False
                    break
            
            # 根据用户反馈调用功能
            update_light("green", False)
            if tag == True:
                text = f"正在为您{speecher_str2[i]}"
                logger.Log(f"典型场景:用户同意{speecher_str2[i]}")
                update_note(text)
                application.schedule(func[i], [])
            elif tag == False:
                text = f"您拒绝{speecher_str2[i]}"
                logger.Log(f"典型场景:用户拒绝{speecher_str2[i]}")
                update_note(text)
                
            time.sleep(1)
        print("退出enter")
        return
