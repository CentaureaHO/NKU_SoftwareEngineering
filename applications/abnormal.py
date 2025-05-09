#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#_author = 'Yidian Lin'

"""
Module Description:
    用于实现异常场景反馈的功能
"""

import os
import time
import requests
import sys
sys.path.append(r'C:\2025spring\软件工程\小组作业\NKU_SoftwareEngineering')

class Abnormal:
    def __init__(self) -> None:
        pass

    @classmethod
    def distraction_reminder(cls,controller) -> None:
        note_text = "注意!请目视前方\n"
        #note_text = "注意!请目视前方\n" + \
        #            "语音输入\"已经注意道路\"解除警告\n" + \
        #            "竖起大拇指确认安全/摇手拒绝警告\n"
        requests.post('http://127.0.0.1:5000/update_string', json={'message': note_text})
        requests.post('http://127.0.0.1:5000/set_blinking', json={'enabled': True})

        Tag = True
        while Tag == True:
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