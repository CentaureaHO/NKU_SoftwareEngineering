#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#_author = 'Yidian Lin'

"""
Module Description:
    用于实现车载多模态智能交互系统的日志功能
"""

import os
import logging

class Logger:
    def __init__(self) -> None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        relative_path = "database/log/Log.log"
        
        # 确保日志目录存在
        log_dir = os.path.join(current_dir, "database/log")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        log_path = os.path.join(current_dir, relative_path)
        self.logger = logging.getLogger("Log")
        self.logger.setLevel(logging.DEBUG)
        
        # 创建文件处理器和控制台处理器
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        console_handler = logging.StreamHandler()
        
        # 创建Formatter（日志格式）
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        # 将Handler添加到Logger
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

    def Log(self, log: str) -> None:
        """
        记录日志信息
        
        Args:
            log: 日志消息
        """
        self.logger.info(log)

logger = Logger()

if __name__ == '__main__':
    # 测试日志功能
    logger.Log("系统初始化")