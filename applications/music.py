#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# _author = 'Yidian Lin'

"""
Module Description:
    用于实现车载音乐相关的功能
"""

import os
import pygame


class Music:
    """音乐管理类"""
    def __init__(self) -> None:
        """构造函数"""
        pygame.mixer.init()
        current_dir = os.path.dirname(os.path.abspath(__file__))
        relative_path = "../database/music"
        self.dir_path = os.path.join(current_dir, relative_path)
        self.paused = False

    def getlist(self) -> str:
        """获取音乐列表"""
        musiclist = []
        for file_name in os.listdir(self.dir_path):
            musiclist.append(file_name)
        return musiclist

    def find(self, music_name: str) -> str:
        """查找音乐文件"""
        for file_name in os.listdir(self.dir_path):
            if music_name not in file_name:
                continue
            file_path = os.path.join(self.dir_path, file_name)
            if os.path.isfile(file_path):
                print(f"File: {file_path}")
                return file_path
        return ""

    def play(self, music_name: str = "") -> None:
        """播放音乐"""
        from viewer.viewer import jump_to_page

        jump_to_page("music")
        file_path = self.find(music_name)
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
        self.paused = False

    def pause(self) -> None:
        """暂停音乐"""
        print("pause")
        if pygame.mixer.music.get_busy() is False:
            self.paused = False
            return

        pygame.mixer.music.pause()
        self.paused = True

    def unpause(self) -> None:
        """恢复音乐"""
        print("unpause")
        if self.paused is False:
            return

        pygame.mixer.music.unpause()
        self.paused = False

    def change_pause(self) -> None:
        """切换音乐播放状态"""
        if self.paused is True:
            self.unpause()
        else:
            self.pause()
