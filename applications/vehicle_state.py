#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# _author = 'Yidian Lin'

"""
Module Description:
    用于实现车辆状态监测相关的功能(由于我们没有实际的车,所以仅做模拟,不真实实现)
"""


class VehicleState:
    """车辆状态监测类:模拟车辆状态监测功能"""

    def __init__(self) -> None:
        """构造函数"""

    def monitor(self):
        """监测车辆状态概述"""
        # 返回 [电量/油量, 胎压状态, 续航里程]
        # return ["80%", "正常", "300km"]
        return [
            self.get_fuel_level(),
            self.get_tire_pressure(),
            self.get_remaining_mileage()
        ]

    def get_fuel_level(self) -> str:
        """获取当前油量/电量 (模拟)"""
        return "80%"

    def get_tire_pressure(self) -> str:
        """获取轮胎压力状态 (模拟)"""
        return "正常"

    def get_remaining_mileage(self) -> str:
        """获取预估续航里程 (模拟)"""
        return "300km"
