"""语音识别配置管理模块，负责管理语音识别的配置项"""

import json
import logging
import os
from typing import Dict, List, Optional

from pypinyin import lazy_pinyin

logger = logging.getLogger('ConfigManager')

# 默认配置
DEFAULT_CONFIG = {
    "wake_words": ["你好小智"],                  # 支持多唤醒词
    "wake_words_pinyin": ["ni hao xiao zhi"],   # 唤醒词拼音
    "enable_wake_word": True,                   # 是否启用唤醒词
    "enable_speaker_verification": True,        # 是否启用声纹识别
    "speaker_verification_threshold": 0.35,     # 声纹识别阈值
    "min_enrollment_duration": 3.0,             # 最小声纹注册时长(秒)
    "max_temp_speakers": 10,                    # 最大临时声纹数量
    "keywords": {                               # 特定关键词分类
        "打开车窗": "window_control",
        "关闭车窗": "window_control",
        "打开空调": "ac_control",
        "关闭空调": "ac_control"
    },
    "exit_keywords": ["再见", "拜拜", "结束", "关闭"]  # 退出聆听状态的关键词
}

class ConfigManager:
    """配置管理类，负责加载和保存语音识别配置"""

    def __init__(self, config_path: str):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> dict:
        """
        加载配置文件，若不存在则创建默认配置
        
        Returns:
            dict: 配置字典
        """
        if not os.path.exists(self.config_path):
            logger.info("配置文件不存在，创建默认配置: %s", self.config_path)
            config_dir = os.path.dirname(self.config_path)
            os.makedirs(config_dir, exist_ok=True)

            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(DEFAULT_CONFIG, f, ensure_ascii=False, indent=4)
            return DEFAULT_CONFIG.copy()

        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                # 检查并补充缺失的配置项
                for key, value in DEFAULT_CONFIG.items():
                    if key not in config:
                        config[key] = value

                # 确保唤醒词和拼音数量一致
                if len(config['wake_words']) != len(config['wake_words_pinyin']):
                    config['wake_words_pinyin'] = [
                        ' '.join(lazy_pinyin(word)) for word in config['wake_words']]

            # 保存更新后的配置
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=4)

            return config
        except (FileNotFoundError, PermissionError) as e:
            logger.error("无法访问配置文件: %s，使用默认配置", e)
            return DEFAULT_CONFIG.copy()
        except json.JSONDecodeError as e:
            logger.error("配置文件格式错误: %s，使用默认配置", e)
            return DEFAULT_CONFIG.copy()
        except OSError as e:
            logger.error("读写配置文件时发生系统错误: %s，使用默认配置", e)
            return DEFAULT_CONFIG.copy()

    def save_config(self) -> bool:
        """
        保存配置到文件
        
        Returns:
            bool: 是否保存成功
        """
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=4)
            return True
        except (FileNotFoundError, PermissionError, OSError) as e:
            logger.error("保存配置文件失败: %s", e)
            return False

    def get_config(self) -> Dict:
        """获取当前配置"""
        return self.config.copy()

    def update_config(self, new_config: Dict) -> bool:
        """
        更新配置
        
        Args:
            new_config: 新的配置字典
            
        Returns:
            bool: 是否更新成功
        """
        try:
            self.config.update(new_config)
            return self.save_config()
        except TypeError as e:
            logger.error("更新配置失败，参数类型错误: %s", e)
            return False
        except (OSError, PermissionError) as e:
            logger.error("更新配置时文件操作失败: %s", e)
            return False

    def add_wake_word(self, wake_word: str) -> bool:
        """
        添加新的唤醒词
        
        Args:
            wake_word: 要添加的唤醒词
            
        Returns:
            bool: 是否添加成功
        """
        if wake_word in self.config["wake_words"]:
            logger.warning("唤醒词 '%s' 已存在", wake_word)
            return False

        wake_pinyin = ' '.join(lazy_pinyin(wake_word))
        self.config["wake_words"].append(wake_word)
        self.config["wake_words_pinyin"].append(wake_pinyin)

        result = self.save_config()
        if result:
            logger.info("添加唤醒词成功: %s (%s)", wake_word, wake_pinyin)
        return result

    def remove_wake_word(self, wake_word: str) -> bool:
        """
        移除唤醒词
        
        Args:
            wake_word: 要移除的唤醒词
            
        Returns:
            bool: 是否移除成功
        """
        if wake_word not in self.config["wake_words"]:
            logger.warning("唤醒词 '%s' 不存在", wake_word)
            return False

        # 至少保留一个唤醒词
        if len(self.config["wake_words"]) <= 1:
            logger.warning("不能移除最后一个唤醒词")
            return False

        idx = self.config["wake_words"].index(wake_word)
        self.config["wake_words"].pop(idx)
        self.config["wake_words_pinyin"].pop(idx)

        result = self.save_config()
        if result:
            logger.info("移除唤醒词成功: %s", wake_word)
        return result

    def toggle_wake_word(self, enable: Optional[bool] = None) -> bool:
        """
        切换唤醒词功能状态
        
        Args:
            enable: 是否启用唤醒词功能，None表示切换
            
        Returns:
            bool: 操作后的唤醒词状态
        """
        if enable is not None:
            self.config["enable_wake_word"] = enable
        else:
            self.config["enable_wake_word"] = not self.config["enable_wake_word"]

        self.save_config()
        logger.info(
            "唤醒词功能已%s",
            "开启" if self.config["enable_wake_word"] else "关闭")
        return self.config["enable_wake_word"]

    def add_keyword(self, keyword: str, category: str) -> bool:
        """
        添加特定关键词
        
        Args:
            keyword: 关键词
            category: 关键词类别
            
        Returns:
            bool: 是否添加成功
        """
        if "keywords" not in self.config:
            self.config["keywords"] = {}

        self.config["keywords"][keyword] = category
        result = self.save_config()

        if result:
            logger.info("添加关键词成功: '%s' -> %s", keyword, category)
        return result

    def remove_keyword(self, keyword: str) -> bool:
        """
        删除特定关键词
        
        Args:
            keyword: 要删除的关键词
            
        Returns:
            bool: 是否删除成功
        """
        if "keywords" not in self.config or keyword not in self.config["keywords"]:
            logger.warning("关键词 '%s' 不存在", keyword)
            return False

        del self.config["keywords"][keyword]
        result = self.save_config()

        if result:
            logger.info("删除关键词成功: '%s'", keyword)
        return result

    def add_exit_keyword(self, keyword: str) -> bool:
        """
        添加退出聆听状态的关键词
        
        Args:
            keyword: 要添加的关键词
            
        Returns:
            bool: 是否添加成功
        """
        if "exit_keywords" not in self.config:
            self.config["exit_keywords"] = ["再见", "拜拜", "结束", "关闭"]

        if keyword in self.config["exit_keywords"]:
            logger.warning("退出关键词 '%s' 已存在", keyword)
            return False

        self.config["exit_keywords"].append(keyword)
        result = self.save_config()

        if result:
            logger.info("添加退出关键词成功: %s", keyword)
        return result

    def remove_exit_keyword(self, keyword: str) -> bool:
        """
        删除退出聆听状态的关键词
        
        Args:
            keyword: 要删除的关键词
            
        Returns:
            bool: 是否删除成功
        """
        if "exit_keywords" not in self.config or keyword not in self.config["exit_keywords"]:
            logger.warning("退出关键词 '%s' 不存在", keyword)
            return False

        # 至少保留一个退出关键词
        if len(self.config["exit_keywords"]) <= 1:
            logger.warning("不能删除最后一个退出关键词")
            return False

        self.config["exit_keywords"].remove(keyword)
        result = self.save_config()

        if result:
            logger.info("删除退出关键词成功: %s", keyword)
        return result

    def get_exit_keywords(self) -> List[str]:
        """
        获取所有退出聆听状态的关键词
        
        Returns:
            List[str]: 退出关键词列表
        """
        return self.config.get("exit_keywords", ["再见", "拜拜", "结束", "关闭"]).copy()
