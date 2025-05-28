"""
Module Description:
    用于存储和管理所有系统组件
"""

# 全局组件字典，存储所有系统组件
_components = {}


def get_component(name):
    """获取已初始化的系统组件"""
    return _components.get(name)


def register_component(name, component):
    """注册系统组件"""
    _components[name] = component
    return component

def get_all_components():
    """获取所有已注册的系统组件"""
    return _components
