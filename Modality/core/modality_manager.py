from typing import Dict, List, Optional, Type
from .base_modality import BaseModality, ModalityState
from .error_codes import (
    SUCCESS, MODALITY_ALREADY_EXISTS, MODALITY_NOT_FOUND,
    MODALITY_START_FAILED, MODALITY_STOP_FAILED
)


class ModalityManager:
    """
    模态管理器类，负责管理所有模态的生命周期和状态
    """

    def __init__(self):
        self._modalities: Dict[str, BaseModality] = {}
        self._active_modalities: List[str] = []

    def register_modality(self, modality: BaseModality) -> int:
        """
        注册一个新的模态

        Args:
            modality: 要注册的模态实例

        Returns:
            int: 错误码，0表示成功，其他值表示失败
        """
        if modality.name in self._modalities:
            return MODALITY_ALREADY_EXISTS

        self._modalities[modality.name] = modality
        return SUCCESS

    def unregister_modality(self, modality_name: str) -> int:
        """
        注销一个模态

        Args:
            modality_name: 要注销的模态名称

        Returns:
            int: 错误码，0表示成功，其他值表示失败
        """
        if modality_name not in self._modalities:
            return MODALITY_NOT_FOUND

        modality = self._modalities[modality_name]
        if modality.is_running():
            result = modality.stop()
            if result != SUCCESS:
                return result

        if modality_name in self._active_modalities:
            self._active_modalities.remove(modality_name)

        del self._modalities[modality_name]
        return SUCCESS

    def start_modality(self, modality_name: str) -> int:
        """
        启动一个模态

        Args:
            modality_name: 要启动的模态名称

        Returns:
            int: 错误码，0表示成功，其他值表示失败
        """
        if modality_name not in self._modalities:
            return MODALITY_NOT_FOUND

        modality = self._modalities[modality_name]
        result = modality.start()

        if result == SUCCESS and modality_name not in self._active_modalities:
            self._active_modalities.append(modality_name)

        return result

    def stop_modality(self, modality_name: str) -> int:
        """
        停止一个模态

        Args:
            modality_name: 要停止的模态名称

        Returns:
            int: 错误码，0表示成功，其他值表示失败
        """
        if modality_name not in self._modalities:
            return MODALITY_NOT_FOUND

        modality = self._modalities[modality_name]
        result = modality.stop()

        if result == SUCCESS and modality_name in self._active_modalities:
            self._active_modalities.remove(modality_name)

        return result

    def get_modality(self, modality_name: str) -> Optional[BaseModality]:
        """
        获取指定名称的模态实例

        Args:
            modality_name: 模态名称

        Returns:
            Optional[BaseModality]: 模态实例，如果不存在则返回None
        """
        return self._modalities.get(modality_name)

    def update_all(self) -> Dict[str, ModalityState]:
        """
        更新所有活跃模态并返回它们的状态

        Returns:
            Dict[str, ModalityState]: 模态名称到状态的映射
        """
        results = {}

        for modality_name in self._active_modalities:
            modality = self._modalities[modality_name]
            state = modality.update()
            if state:
                results[modality_name] = state

        return results

    def shutdown_all(self) -> int:
        """
        关闭所有模态

        Returns:
            int: 错误码，0表示成功，其他值表示有一个或多个模态关闭失败
        """
        overall_result = SUCCESS

        for modality_name in list(self._active_modalities):
            result = self.stop_modality(modality_name)
            if result != SUCCESS:
                overall_result = MODALITY_STOP_FAILED

        return overall_result

    def get_all_modalities(self) -> Dict[str, BaseModality]:
        """
        获取所有注册的模态

        Returns:
            Dict[str, BaseModality]: 模态名称到实例的映射
        """
        return self._modalities.copy()

    def get_active_modalities(self) -> List[str]:
        """
        获取所有活跃模态的名称

        Returns:
            List[str]: 活跃模态名称列表
        """
        return self._active_modalities.copy()

    def get_all_key_info(self) -> dict[str, str]:
        """
        获取所有活跃模态的关键信息

        Returns:
            dict[str, str]: 所有活跃模态的关键信息
        """
        results = {}

        for modality_name in self._active_modalities:
            print(modality_name)
            modality = self._modalities[modality_name]
            key_info = modality.get_key_info()
            results[modality_name] = key_info

        return results
