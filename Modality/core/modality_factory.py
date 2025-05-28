from typing import Dict, Optional, Type

from .base_modality import BaseModality


class ModalityFactory:
    """
    模态工厂类，负责创建不同类型的模态实例
    """

    _registry: Dict[str, Type[BaseModality]] = {}

    @classmethod
    def register_modality_type(cls, modality_type: str, modality_class: Type[BaseModality]) -> None:
        """
        注册一个模态类型

        Args:
            modality_type: 模态类型名称
            modality_class: 模态类的类型
        """
        cls._registry[modality_type] = modality_class

    @classmethod
    def create_modality(cls, modality_type: str, name: str, **kwargs) -> Optional[BaseModality]:
        """
        创建一个模态实例

        Args:
            modality_type: 模态类型名称
            name: 模态实例名称
            **kwargs: 传递给模态构造函数的其他参数

        Returns:
            Optional[BaseModality]: 创建的模态实例，如果类型不存在则返回None
        """
        if modality_type not in cls._registry:
            return None

        modality_class = cls._registry[modality_type]
        return modality_class(name=name, **kwargs)

    @classmethod
    def get_available_types(cls) -> Dict[str, Type[BaseModality]]:
        """
        获取所有可用的模态类型

        Returns:
            Dict[str, Type[BaseModality]]: 模态类型名称到类的映射
        """
        return cls._registry.copy()
