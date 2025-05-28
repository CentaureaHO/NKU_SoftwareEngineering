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
        pass
    #     current_dir = os.path.dirname(os.path.abspath(__file__))
    #     relative_path = "../database/navigation"
    #     self.dir_path = os.path.join(current_dir, relative_path)

    # def find(self,src: str,dest :str) -> str:
    #     file_name = src + "_" + dest + ".jpg"
    #     file_path = os.path.join(self.dir_path, file_name)
    #     return file_path

    # def show(self,src: str,dest :str) -> None:
    #     file_path = self.find(src,dest)
    #     # TODO():异常处理
    #     with Image.open(file_path) as img:
    #         img.show()

    # def getlist(self) -> str:
    #     navigation_list = os.listdir(self.dir_path)
    #     navigation_list = [file for file in navigation_list if file.endswith('.jpg') and file != "None.jpg"] 
    #     navigation_list = [file[:-4] for file in navigation_list]
    #     print("获取导航列表:", navigation_list)
    #     return navigation_list

    def navigate(self,from_ = None,to_ = None) -> None:
        print(f"开始导航从 {from_} 到 {to_}")
        from viewer.viewer import navigate as navigate_ ,jump_to_page
        if(from_ is None) or (to_ is None):
            navigate_()
        else:
            navigate_(from_, to_)
        jump_to_page('navigation')

if __name__ == '__main__':
    navigation = Navigation()
    navigation.show("南开大学津南校区","南开大学八里台校区")