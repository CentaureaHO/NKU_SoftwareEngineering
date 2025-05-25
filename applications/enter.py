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


class Enter:
    def __init__(self) -> None:
        pass

    def enter(self,controller) -> None:
        text = "系统初始化完毕，请驾驶员目视前方"
        from viewer.viewer import update_note, update_light
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
        
        text = "驾驶员已就位"
        update_note(text)
        update_light("green", False)
        time.sleep(1)
        speecher_player.speech_synthesize_sync(text)
        time.sleep(2)

        text = "是否为您播放音乐"
        speecher_player.speech_synthesize_sync(text)
        text = "是否为您播放音乐\n" + \
               "同意请语音输入\"同意播放音乐\"、竖起大拇指或点头\n" \
               "拒绝请语音输入\"拒绝播放音乐\"、摇手或摇头\n"
        update_note(text)

        tag = None
        while tag == None:
            # 获取语音输入
            state = controller.speecher.get_key_info()
            if state != None:
                print("语音输入:",state)
            if state == "同意播放音乐":
                tag = True
                break
            elif state == "拒绝播放音乐":
                tag = False
                break
            # 获取手势输入
            state = controller.static_gesture_tracker.get_key_info()
            if state != None:
                print("手势输入:",state)
            if state == "竖起大拇指":
                tag = True
                break
            elif state == "摇手":
                tag = False
                break
            # 获取头部姿态输入
            state = controller.headposer.get_key_info()
            if state != None:
                print("手势输入:",state)
            if state == "点头":
                tag = True
                break
            elif state == "摇头":
                tag = False
                break
        print("退出enter")
        return

        Tag = True
        while Tag == True:
            if time.time() - start_time > 5:
                speecher_player.speech_synthesize_sync(note_speaker_text2)
            states = controller.manager.update_all()
            # print(states)
            for name, state in states.items():
                # print(f"模态:{name}")
                if name == "speech_recognition":
                    if state and state.recognition["text"]:
                        text = state.recognition["text"]
                        print(f"识别结果: {text}")
                        if text == "已经注意道路":
                            requests.post('http://127.0.0.1:5000/update_string', json={'message': "解除警告"})
                            Tag = False
                            break
                elif name == "static_gesture_tracker":
                    if state.detections["gesture"]["detected"]:
                        name = state.detections["gesture"]["name"]
                        print(f"手势识别结果: {name}")
                        if name == "5":
                            requests.post('http://127.0.0.1:5000/update_string', json={'message': "拒绝警告"})
                            Tag = False
                            break
                        elif name == "6":
                            requests.post('http://127.0.0.1:5000/update_string', json={'message': "确认安全"})
                            Tag = False
                            break
        
        requests.post('http://127.0.0.1:5000/set_blinking', json={'enabled': False})


if __name__ == '__main__':
    pass