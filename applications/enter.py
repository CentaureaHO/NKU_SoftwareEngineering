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
        #start_time = time.time()  # 记录开始时间
        text = "系统初始化完毕，请驾驶员目视前方"
        #note_speaker_text1 = "请注意行车安全"
        #note_speaker_text2 = "请立即目视前方"
        #note_text = "注意!请目视前方\n" + \
        #            "语音输入\"已经注意道路\"解除警告\n" + \
        #            "竖起大拇指确认安全/摇手拒绝警告\n"
        from viewer.viewer import update_note, update_light
        update_note(text)
        update_light("red", True)
        while True:
            state = controller.gazer.get_key_info()
            print(f"获取到的状态: {state}")
            # start_time = time.time()  # 记录开始时间

        # while True:
        #     update_light("red", True)
        #     time.sleep(10)
        #     update_light("green", False)
        #     time.sleep(10)
        #     update_light("red", False)
        #     time.sleep(10)
        #     update_light("green", True)
        #requests.post('http://127.0.0.1:5000/update_string', json={'message': note_text})
        #requests.post('http://127.0.0.1:5000/set_blinking', json={'enabled': True})
        speecher_player.speech_synthesize_sync(text)
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