#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#_author = 'Yidian Lin'

"""
Module Description:
    用于实现车载导航相关的功能
"""

import os
from PIL import Image

class Navigation:
    def __init__(self) -> None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        relative_path = "../database/navigation"
        self.dir_path = os.path.join(current_dir, relative_path)

    def find(self,src: str,dest :str) -> str:
        file_name = src + "_" + dest + ".jpg"
        file_path = os.path.join(self.dir_path, file_name)
        return file_path

    def show(self,src: str,dest :str) -> None:
        file_path = self.find(src,dest)
        # TODO():异常处理
        with Image.open(file_path) as img:
            img.show()

if __name__ == '__main__':
    navigation = Navigation()
    navigation.show("南开大学津南校区","南开大学八里台校区")