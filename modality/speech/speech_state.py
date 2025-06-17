"""语音状态模块，提供语音识别状态的数据结构和处理方法"""

from typing import Any, Dict
from dataclasses import dataclass

from modality.core.base_modality import ModalityState


@dataclass
class SpeechState(ModalityState):
    """语音状态类，继承自ModalityState"""

    def __init__(self, timestamp: float = None):
        super().__init__(timestamp)
        # 语音识别结果
        self.recognition = {
            "text": "",               # 识别的原始文本
            "wake_word": "",          # 触发的唤醒词
            "has_wake_word": False,   # 是否包含唤醒词
            "keyword": "",            # 识别到的关键词
            "keyword_category": "",   # 关键词所属类别
            "has_keyword": False,     # 是否包含特定关键词
            "is_command": False,      # 是否是指令(包含唤醒词或关键词)
            "speaker_id": "",         # 说话人ID
            "speaker_name": "",       # 说话人名称
            "is_registered_speaker": False,  # 是否是注册过的说话人
            "confidence": 0.0         # 识别置信度
        }

    def to_dict(self) -> Dict[str, Any]:
        """将状态转换为字典"""
        result = super().to_dict()
        result.update({"recognition": self.recognition})
        return result
