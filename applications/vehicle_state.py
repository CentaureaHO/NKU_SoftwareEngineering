#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#_author = 'Yidian Lin'

"""
Module Description:
    用于实现车辆状态监测相关的功能
"""

from enum import Enum

class VehicleState:
    type = Enum("type", ["oil_quantity","tire_pressure","mileage"])
    def __init__(self) -> None:
        pass

    def monitor(self,state_type: Enum) -> str:
        if state_type == VehicleState.type.oil_quantity:
            return "当前剩余油量为100%"
        elif state_type == VehicleState.type.tire_pressure:
            return "当前胎压正常"
        elif state_type == VehicleState.type.mileage:
            return "当前总行驶里程为0km"

if __name__ == '__main__':
    state = VehicleState()
    print(state.monitor(VehicleState.type.oil_quantity))
    print(state.monitor(VehicleState.type.tire_pressure))
    print(state.monitor(VehicleState.type.mileage))