#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#_author = 'Yidian Lin'

"""
Module Description:
    用于实现车载音乐相关的功能
"""

import os
import time
import pygame


# 在 Flask 运行时的任意地方调用这个接口
import requests


class Music:
    def __init__(self) -> None:
        pygame.mixer.init()
        current_dir = os.path.dirname(os.path.abspath(__file__))
        relative_path = "../database/music"
        self.dir_path = os.path.join(current_dir, relative_path)
        self.paused = False

    def getlist(self) -> str:
        musiclist = []
        for file_name in os.listdir(self.dir_path):
            musiclist.append(file_name)
        return musiclist

    def find(self,music_name: str) -> str:
        for file_name in os.listdir(self.dir_path):
            if music_name not in file_name:
                continue
            file_path = os.path.join(self.dir_path, file_name)
            if os.path.isfile(file_path):
                print(f"File: {file_path}")
                return file_path
        return ""

    def play(self,music_name: str) -> None:
        file_path = self.find(music_name)
        # TODO():异常处理
        pygame.mixer.music.load(file_path)  # 加载文件
        pygame.mixer.music.play()  # 开始播放
        self.paused = False
        #while pygame.mixer.music.get_busy():
        #    pygame.time.Clock().tick(10)

    def pause(self) -> None:
        print("pause")
        if pygame.mixer.music.get_busy() == False:
            self.paused = False
            return
        
        pygame.mixer.music.pause()
        self.paused = True

    def unpause(self) -> None:
        print("unpause")
        if self.paused == False:
            return
        
        pygame.mixer.music.unpause()
        self.paused = False

    def change_pause(self) -> None:
        if self.paused == True:
            self.unpause()
        else:
            self.pause()
            

if __name__ == '__main__':
    music = Music()
    print(music.getlist())
    music.play("T1P2")
    time.sleep(5)
    music.change_pause()
    time.sleep(5)
    music.change_pause()
    time.sleep(5)
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)