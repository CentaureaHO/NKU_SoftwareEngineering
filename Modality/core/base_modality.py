from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import numpy as np
import time
from .error_codes import SUCCESS, ALREADY_INITIALIZED, NOT_INITIALIZED, OPERATION_FAILED

class ModalityState:
    def __init__(self, timestamp: float = None):
        self.timestamp = timestamp or time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        return {"timestamp": self.timestamp}


class BaseModality(ABC):
    def __init__(self, name: str):
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
        pass
    
    @abstractmethod
    def update(self) -> Optional[ModalityState]:
        """
        更新模态状态
        
        Returns:
            Optional[ModalityState]: 更新后的模态状态，如果失败则返回None
        """
        pass
    
    @abstractmethod
    def shutdown(self) -> int:
        """
        关闭模态
        
        Returns:
            int: 错误码，0表示成功，其他值表示失败
        """
        pass
    
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
