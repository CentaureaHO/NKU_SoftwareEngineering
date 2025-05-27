#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#_author = 'Yidian Lin'

"""
Module Description:
    用于实现车载导航相关的功能
"""


class Navigation:
    """车载导航"""
    def __init__(self) -> None:
        """构造函数"""

    def navigate(self,from_ = None,to_ = None) -> None:
        """导航"""
        print(f"开始导航从 {from_} 到 {to_}")
        from viewer.viewer import navigate as navigate_ ,jump_to_page
        if(from_ is None) or (to_ is None):
            navigate_()
        else:
            navigate_(from_, to_)
        jump_to_page('navigation')
