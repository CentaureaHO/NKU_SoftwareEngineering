#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# _author = 'Xianda Tang' 'Yidian Lin'

"""
Module Description:
    用于存储所有系统组件
"""

import os
import time


class Setting:
    "基于声纹的权限控制类"

    def __init__(self, speecher):
        "构造函数"
        self.speecher = speecher
        current_dir = os.path.dirname(os.path.abspath(__file__))
        relative_path = "./database/setting/setting.txt"
        self.file_path = os.path.join(current_dir, relative_path)
        with open(self.file_path, 'r', encoding="utf-8") as file:
            self.driver = file.read()

    def register_voiceprint(self, username):
        "注册声纹"
        print(f"注册新的声纹，用户名为：{username}")
        self.speecher.register_speaker(username)
        time.sleep(10)
        state = self.speecher.update()
        if state and state.recognition["text"]:
            print(f"识别结果: {state.recognition['text']}")

    def delete_voiceprint(self, username):
        "删除声纹"
        print(f"删除声纹，用户名为：{username}")
        idx = int(username.split('.')[0]) - 1
        speakers = self.speecher.get_registered_speakers()
        if 0 <= idx < len(speakers):
            speaker_id = list(speakers.keys())[idx]
            if self.speecher.delete_speaker(speaker_id):
                print(f"已删除声纹 {speakers[speaker_id]['name']}")
        else:
            print("无效的编号")

    def get_voiceprints(self):
        "获取已注册声纹列表"
        voiceprints = []
        print("查看注册声纹列表")
        speakers = self.speecher.get_registered_speakers()
        if not speakers:
            print("没有已注册的声纹")
            return voiceprints

        print("已注册声纹:")
        for idx, (speaker_id, info) in enumerate(speakers.items()):
            print(f"  {idx+1}. {info['name']} (ID: {speaker_id})")
            voiceprints.append(f"{idx+1}. {info['name']} (ID: {speaker_id})")
        return voiceprints

    def set_driver(self, driver_name):
        "将指定声纹设置为驾驶员"
        print(f"设置驾驶员为: {driver_name}")
        speakers = self.speecher.get_registered_speakers()
        idx = int(driver_name.split('.')[0]) - 1
        if 0 <= idx < len(speakers):
            speaker_id = list(speakers.keys())[idx]
        else:
            print("无效的编号")
            return
        print(f"将声纹 {speakers[speaker_id]['name']}设为驾驶员")
        self.driver = speakers[speaker_id]['name']
        with open(self.file_path, 'w', encoding="utf-8") as file:
            file.write(self.driver)

    def get_driver(self):
        "获取驾驶员"
        print("驾驶员信息:", self.driver)
        return self.driver
