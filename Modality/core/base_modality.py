"""基本模态模块，定义了模态的基类和状态类。"""
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from dataclasses import dataclass

from .error_codes import (ALREADY_INITIALIZED, NOT_INITIALIZED,
                          SUCCESS)


@dataclass
class ModalityState:
    """表示模态在特定时间点的状态。"""
    def __init__(self, timestamp: float = None):
        """
        初始化 ModalityState。

        Args:
            timestamp (float, optional): 状态的时间戳。默认为当前时间。
        """
        self.timestamp = timestamp or time.time()

    def to_dict(self) -> Dict[str, Any]:
        """将模态状态转换为字典。

        Returns:
            Dict[str, Any]: 包含时间戳的字典。
        """
        return {"timestamp": self.timestamp}


class BaseModality(ABC):
    """所有具体模态类的抽象基类。"""
    def __init__(self, name: str):
        """
        初始化 BaseModality。

        Args:
            name (str): 模态的名称。
        """
        self.name = name
        self._is_running = False
        self._last_state: Optional[ModalityState] = None

    @abstractmethod
    def initialize(self) -> int:
        """
        初始化模态

        Returns:
            int: 错误码，0表示成功，其他值表示失败
        """

    @abstractmethod
    def update(self) -> Optional[ModalityState]:
        """
        更新模态状态

        Returns:
            Optional[ModalityState]: 更新后的模态状态，如果失败则返回None
        """

    @abstractmethod
    def shutdown(self) -> int:
        """
        关闭模态

        Returns:
            int: 错误码，0表示成功，其他值表示失败
        """

    def start(self) -> int:
        """
        启动模态

        Returns:
            int: 错误码，0表示成功，其他值表示失败
        """
        if self._is_running:
            return ALREADY_INITIALIZED

        result = self.initialize()
        if result == SUCCESS:
            self._is_running = True

        return result

    def stop(self) -> int:
        """
        停止模态

        Returns:
            int: 错误码，0表示成功，其他值表示失败
        """
        if not self._is_running:
            return NOT_INITIALIZED

        result = self.shutdown()
        if result == SUCCESS:
            self._is_running = False

        return result

    def get_last_state(self) -> Optional[ModalityState]:
        """
        获取最后一次状态

        Returns:
            Optional[ModalityState]: 最后一次的状态
        """
        return self._last_state

    def is_running(self) -> bool:
        """
        检查模态是否正在运行

        Returns:
            bool: 如果模态正在运行则返回True，否则返回False
        """
        return self._is_running

    # @abstractmethod
    def get_key_info(self) -> str:
        """
        获取模态的关键信息

        Returns:
            str: 模态的关键信息
        """

        return "Not implemented in base class"
