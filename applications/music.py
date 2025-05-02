#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#_author = 'Yidian Lin'

"""
Module Description:
    用于实现车载音乐相关的功能
"""

import os
import pygame

class Music:
    def __init__(self) -> None:
        pygame.mixer.init()
        current_dir = os.path.dirname(os.path.abspath(__file__))
        relative_path = "../database/music"
        self.dir_path = os.path.join(current_dir, relative_path)

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
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

    def pause() -> None:
        pygame.mixer.music.pause()

    def unpause() -> None:
        pygame.mixer.music.unpause()

if __name__ == '__main__':
    music = Music()
    music.play("南开校歌")