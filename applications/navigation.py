#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# _author = 'Yidian Lin'

"""
Module Description:
    用于实现车载导航相关的功能
"""

# 解决 C0415: Import outside toplevel
from viewer.viewer import jump_to_page
from viewer.viewer import navigate as navigate_


class Navigation:
    """车载导航"""

    def __init__(self) -> None:
        """构造函数"""
        self.from_ = None
        self.to_ = None

    def navigate(self, from_=None, to_=None) -> None:
        """导航"""
        print(f"开始导航从 {from_} 到 {to_}")
        if (from_ is None) or (to_ is None):
            navigate_()
        else:
            navigate_(from_, to_)
        self.from_ = from_
        self.to_ = to_
        jump_to_page('navigation')

    def get_navigation_list(self) -> list:
        """获取导航列表"""
        return (self.from_, self.to_)
