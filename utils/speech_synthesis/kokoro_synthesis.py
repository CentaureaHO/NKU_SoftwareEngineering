import os
import time
import json
import logging
import shutil
import asyncio
import numpy as np
import soundfile as sf
import torch
from pathlib import Path
from typing import Dict, List, Optional, Union, Generator, Callable, Tuple, Any
from threading import Lock
from kokoro import KModel, KPipeline
# 替换 modelscope 为 huggingface_hub
from huggingface_hub import snapshot_download as hf_snapshot_download

logging.basicConfig(
    level=logging.DEBUG if os.environ.get('SYNTHESIS_DEBUG', '0') == '1' else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='kokoro_synthesis.log',
    filemode='w'
)
logger = logging.getLogger('KokoroSynthesis')

REPO_ID = 'hexgrad/Kokoro-82M-v1.1-zh'
MODEL_NAME = 'kokoro-v1_1-zh.pth'
CONFIG_NAME = 'config.json'
SAMPLE_RATE = 24000
DEFAULT_VOICE = 'zf_001'

DEFAULT_CONFIG = {
    "model_path": "",
    "voices_dir": "",
    "default_voice": DEFAULT_VOICE,
    "custom_pronunciations": {
        "Kokoro": "kˈOkəɹO",
        "Sol": "sˈOl"
    },
    "speed_factor": 1.0
}

class Singleton(type):
    """单例模式元类"""
    _instances = {}
    _locks = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            if cls not in cls._locks:
                cls._locks[cls] = Lock()
            with cls._locks[cls]:
                if cls not in cls._instances:
                    cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class KokoroSynthesis(metaclass=Singleton):
    """
    Kokoro语音合成单例类，提供流式TTS功能
    """
    
    def __init__(self, model_dir: str = None, config_path: str = None):
        """
        初始化Kokoro语音合成器
        
        Args:
            model_dir: 模型目录路径，默认为utils/models/kokoro
            config_path: 配置文件路径，默认在model_dir下
        """
        # 设置模型目录
        if model_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.model_dir = os.path.join(base_dir, "models", "kokoro")
        else:
            self.model_dir = model_dir
            
        self.voices_dir = os.path.join(self.model_dir, "voices")
        self.model_cache_dir = os.path.join(self.model_dir, "model_cache")
        
        # 创建必要的目录
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.voices_dir, exist_ok=True)
        os.makedirs(self.model_cache_dir, exist_ok=True)

        # 设置配置文件路径
        if config_path is None:
            self.config_path = os.path.join(self.model_dir, "synthesis_config.json")
        else:
            self.config_path = config_path
            
        # 加载配置
        self.config = self._load_config()
        
        # 初始化模型和管道相关变量
        self.zh_model = None
        self.zh_pipeline = None
        self.en_pipeline = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.is_initialized = False
        
        # 当前话者
        self.current_voice = self.config.get("default_voice", DEFAULT_VOICE)
        
    def _load_config(self) -> dict:
        """加载配置文件，若不存在则创建默认配置"""
        if not os.path.exists(self.config_path):
            logger.info(f"配置文件不存在，创建默认配置: {self.config_path}")
            default_config = DEFAULT_CONFIG.copy()
            default_config["voices_dir"] = self.voices_dir
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, ensure_ascii=False, indent=4)
            return default_config
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                
                # 确保所有必要的配置项都存在
                for key, value in DEFAULT_CONFIG.items():
                    if key not in config:
                        config[key] = value
                        
                # 更新voices_dir
                if not config.get("voices_dir"):
                    config["voices_dir"] = self.voices_dir
                
            # 保存更新后的配置
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=4)
                
            return config
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}，使用默认配置")
            default_config = DEFAULT_CONFIG.copy()
            default_config["voices_dir"] = self.voices_dir
            return default_config
    
    def _save_config(self) -> bool:
        """保存当前配置到配置文件"""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=4)
            return True
        except Exception as e:
            logger.error(f"保存配置文件失败: {e}")
            return False
    
    def _download_and_cache_model(self, model_id: str, model_version: str = None) -> str:
        """
        使用huggingface_hub下载并缓存模型，如果本地已有缓存则直接使用本地模型
        
        Args:
            model_id: 模型ID
            model_version: 模型版本（可选）
            
        Returns:
            str: 模型本地缓存路径
        """
        cache_key = f"{model_id.replace('/', '_')}"
        if model_version:
            cache_key += f"_{model_version}"
        
        model_cache_path = os.path.join(self.model_cache_dir, cache_key)
        model_info_path = os.path.join(model_cache_path, "model_info.json")
        model_lock_path = os.path.join(model_cache_path, ".lock")
        
        if os.path.exists(model_cache_path) and os.path.exists(model_info_path):
            try:
                with open(model_info_path, 'r', encoding='utf-8') as f:
                    model_info = json.load(f)
                logger.info(f"模型已存在本地: {model_id} ({model_info.get('timestamp')})")
                return model_info.get('path', model_cache_path)
            except Exception as e:
                logger.warning(f"读取本地模型信息失败: {e}")
                if not os.path.exists(model_lock_path):
                    try:
                        logger.info(f"尝试清理损坏的模型缓存: {model_cache_path}")
                        shutil.rmtree(model_cache_path, ignore_errors=True)
                        os.makedirs(model_cache_path, exist_ok=True)
                    except Exception as cleanup_err:
                        logger.error(f"清理损坏的模型缓存失败: {cleanup_err}")
        
        if os.path.exists(model_lock_path):
            lock_modified_time = os.path.getmtime(model_lock_path)
            if time.time() - lock_modified_time > 1800:  # 30分钟
                logger.warning(f"发现过期的锁文件，可能是之前的下载异常终止")
                try:
                    os.remove(model_lock_path)
                except Exception:
                    logger.error(f"无法删除过期的锁文件: {model_lock_path}")
                    raise RuntimeError(f"模型缓存目录被锁定，且无法释放锁。请手动删除: {model_lock_path}")
            else:
                logger.warning(f"另一个进程正在下载模型，等待下载完成...")
                for _ in range(60):
                    time.sleep(10)
                    if not os.path.exists(model_lock_path):
                        break
                if os.path.exists(model_lock_path):
                    raise RuntimeError(f"等待其他进程下载模型超时。如果确定没有其他下载正在进行，请手动删除锁文件: {model_lock_path}")
                
                if os.path.exists(model_info_path):
                    try:
                        with open(model_info_path, 'r', encoding='utf-8') as f:
                            model_info = json.load(f)
                        logger.info(f"模型已由另一进程下载完成: {model_id}")
                        return model_info.get('path', model_cache_path)
                    except Exception:
                        pass

        os.makedirs(model_cache_path, exist_ok=True)
        try:
            with open(model_lock_path, 'w') as lock_file:
                lock_file.write(f"PID: {os.getpid()}, Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
                
            try:
                logger.info(f"正在从Hugging Face下载模型: {model_id} {model_version if model_version else ''}")
                # 使用huggingface_hub的snapshot_download函数
                model_path = hf_snapshot_download(
                    repo_id=model_id, 
                    revision=model_version,
                    cache_dir=model_cache_path,
                    local_dir=os.path.join(model_cache_path, "model"),
                    local_dir_use_symlinks=False
                )
                
                model_info = {
                    "model_id": model_id,
                    "version": model_version,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "path": model_path
                }
                
                with open(model_info_path, 'w', encoding='utf-8') as f:
                    json.dump(model_info, f, ensure_ascii=False, indent=4)
                    
                logger.info(f"模型下载成功: {model_id}")
                return model_path
                
            except Exception as e:
                logger.error(f"从Hugging Face下载模型失败: {e}")
                try:
                    incomplete_marker = os.path.join(model_cache_path, ".incomplete")
                    with open(incomplete_marker, 'w') as f:
                        f.write(f"Download failed: {str(e)}")
                except Exception:
                    pass
                raise
            finally:
                try:
                    if os.path.exists(model_lock_path):
                        os.remove(model_lock_path)
                except Exception as del_err:
                    logger.error(f"删除锁文件失败: {del_err}")
        except Exception as lock_err:
            logger.error(f"创建锁文件失败: {lock_err}")
            raise RuntimeError(f"无法创建模型下载锁，可能没有写入权限: {model_lock_path}")
    
    def _copy_voices_from_repo(self, repo_dir: str) -> bool:
        """从下载的模型仓库中复制声音文件到voices目录"""
        try:
            repo_voices_dir = os.path.join(repo_dir, "voices")
            if os.path.exists(repo_voices_dir) and os.path.isdir(repo_voices_dir):
                voice_files = [f for f in os.listdir(repo_voices_dir) if f.endswith('.pt')]
                
                if not voice_files:
                    logger.warning(f"仓库中未找到声音文件: {repo_voices_dir}")
                    return False
                
                for voice_file in voice_files:
                    src_path = os.path.join(repo_voices_dir, voice_file)
                    dst_path = os.path.join(self.voices_dir, voice_file)
                    
                    if not os.path.exists(dst_path):
                        shutil.copy2(src_path, dst_path)
                        logger.info(f"复制声音文件: {voice_file}")
                
                return True
            else:
                logger.warning(f"仓库中不存在声音目录: {repo_voices_dir}")
                return False
        except Exception as e:
            logger.error(f"复制声音文件失败: {e}")
            return False
    
    def _create_en_callable(self) -> Callable:
        """创建英文发音回调函数"""
        custom_pronunciations = self.config.get("custom_pronunciations", {})
        
        def en_callable(text):
            if text in custom_pronunciations:
                return custom_pronunciations[text]
            return next(self.en_pipeline(text)).phonemes
        
        return en_callable
    
    def _speed_callable(self, len_ps) -> float:
        """
        创建语速调整回调函数
        对于较长的文本，减慢语速以提高质量
        """
        base_speed = self.config.get("speed_factor", 1.0)
        
        # 简单的分段线性函数，随着文本长度增加减慢语速
        if len_ps <= 83:
            speed = 1
        elif len_ps < 183:
            speed = 1 - (len_ps - 83) / 500
        else:
            speed = 0.8
            
        return speed * base_speed
    
    async def initialize(self) -> bool:
        """异步初始化模型和管道"""
        if self.is_initialized:
            logger.info("模型已初始化，无需重复初始化")
            return True
            
        try:
            logger.info(f"正在初始化Kokoro语音合成模型，使用设备: {self.device}")
            
            # 下载或加载模型
            try:
                model_path = self._download_and_cache_model(REPO_ID)
                
                # 设置模型和配置路径
                if model_path:
                    # 处理huggingface下载的文件路径
                    model_dir = model_path
                    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
                    config_files = [f for f in os.listdir(model_dir) if f == 'config.json']
                    
                    if not model_files:
                        raise FileNotFoundError(f"在下载的模型目录中未找到模型文件: {model_dir}")
                    
                    if not config_files:
                        raise FileNotFoundError(f"在下载的模型目录中未找到配置文件: {model_dir}")
                    
                    model_file_path = os.path.join(model_dir, model_files[0])
                    config_file_path = os.path.join(model_dir, config_files[0])
                    
                    # 保存模型路径到配置
                    self.config["model_path"] = model_file_path
                    self._save_config()
                    
                    # 复制声音文件到voices目录
                    self._copy_voices_from_repo(model_dir)
                    
                    # 加载模型
                    self.zh_model = KModel(
                        repo_id=REPO_ID,
                        config=config_file_path,
                        model=model_file_path
                    ).to(self.device).eval()
                    
                    logger.info("Kokoro语音模型加载成功")
                else:
                    raise RuntimeError("模型下载失败")
                    
            except Exception as e:
                logger.error(f"模型下载或加载失败: {e}")
                # 尝试从本地加载已保存的模型路径
                if self.config.get("model_path") and os.path.exists(self.config["model_path"]):
                    model_dir = os.path.dirname(self.config["model_path"])
                    config_file_path = os.path.join(model_dir, CONFIG_NAME)
                    
                    logger.info(f"尝试从本地路径加载模型: {self.config['model_path']}")
                    self.zh_model = KModel(
                        repo_id=REPO_ID,
                        config=config_file_path,
                        model=self.config["model_path"]
                    ).to(self.device).eval()
                    
                    logger.info("从本地路径加载Kokoro语音模型成功")
                else:
                    raise
            
            # 初始化管道
            self.en_pipeline = KPipeline(lang_code='a', repo_id=REPO_ID, model=False)
            en_callable = self._create_en_callable()
            
            # 初始化中文管道
            self.zh_pipeline = KPipeline(
                lang_code='z', 
                repo_id=REPO_ID, 
                model=self.zh_model,
                en_callable=en_callable
            )
            
            # 检查语音文件
            self._ensure_voices_exist()
            
            self.is_initialized = True
            logger.info("Kokoro语音合成系统初始化完成")
            
            return True
            
        except Exception as e:
            logger.error(f"初始化失败: {e}")
            return False
    
    def _ensure_voices_exist(self) -> bool:
        """确保存在可用的语音文件，不存在则下载"""
        voices = self.get_available_voices()
        
        if not voices:
            logger.warning("未找到可用的声音文件，尝试重新下载模型并提取声音文件")
            try:
                model_path = self._download_and_cache_model(REPO_ID)
                if model_path:
                    self._copy_voices_from_repo(model_path)
                    
                    # 检查是否成功复制声音文件
                    voices = self.get_available_voices()
                    if not voices:
                        logger.error("声音文件下载失败")
                        return False
                    return True
                else:
                    logger.error("模型下载失败，无法提取声音文件")
                    return False
            except Exception as e:
                logger.error(f"声音文件获取失败: {e}")
                return False
        return True
    
    async def synthesize(self, text: str, voice: str = None, speed: float = None) -> np.ndarray:
        """
        异步合成语音
        
        Args:
            text: 要合成的文本
            voice: 要使用的声音名称，None使用默认声音
            speed: 语速，None使用动态语速
            
        Returns:
            np.ndarray: 合成的音频数据
        """
        if not self.is_initialized:
            await self.initialize()
            
        if not self.is_initialized:
            raise RuntimeError("语音合成系统初始化失败")
            
        # 确定使用的声音
        voice_name = voice or self.current_voice
        if voice_name != self.current_voice:
            self.set_current_voice(voice_name)
            
        # 确定语音文件路径
        voice_path = os.path.join(self.voices_dir, f"{voice_name}.pt")
        if not os.path.exists(voice_path):
            available_voices = self.get_available_voices()
            if not available_voices:
                raise FileNotFoundError("未找到可用的声音文件")
            voice_name = available_voices[0]
            voice_path = os.path.join(self.voices_dir, f"{voice_name}.pt")
            logger.warning(f"指定的声音文件不存在，使用可用的第一个声音: {voice_name}")
            
        # 处理语速
        speed_callable = None
        if speed is not None:
            speed_callable = lambda _: speed
        else:
            speed_callable = self._speed_callable
            
        try:
            logger.debug(f"开始合成文本: '{text}', 使用声音: {voice_name}")
            generator = self.zh_pipeline(text, voice=voice_path, speed=speed_callable)
            result = next(generator)
            logger.debug("语音合成完成")
            return result.audio
        except Exception as e:
            logger.error(f"语音合成失败: {e}")
            raise
    
    async def synthesize_to_file(self, text: str, output_file: str, voice: str = None, speed: float = None) -> str:
        """
        异步合成语音并保存到文件
        
        Args:
            text: 要合成的文本
            output_file: 输出文件路径
            voice: 要使用的声音名称
            speed: 语速
            
        Returns:
            str: 生成的音频文件路径
        """
        audio_data = await self.synthesize(text, voice, speed)
        
        try:
            # 确保输出目录存在
            output_dir = os.path.dirname(output_file)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                
            sf.write(output_file, audio_data, SAMPLE_RATE)
            logger.info(f"语音已保存到文件: {output_file}")
            return output_file
        except Exception as e:
            logger.error(f"保存语音文件失败: {e}")
            raise
    
    def get_available_voices(self) -> List[str]:
        """
        获取可用的声音列表
        
        Returns:
            List[str]: 可用声音名称列表
        """
        voices = []
        try:
            if not os.path.exists(self.voices_dir):
                os.makedirs(self.voices_dir, exist_ok=True)
                
            voice_files = [f for f in os.listdir(self.voices_dir) if f.endswith('.pt')]
            voices = [os.path.splitext(f)[0] for f in voice_files]
            return voices
        except Exception as e:
            logger.error(f"获取可用声音列表失败: {e}")
            return []
    
    def set_current_voice(self, voice_name: str) -> bool:
        """
        设置当前使用的声音
        
        Args:
            voice_name: 声音名称
            
        Returns:
            bool: 是否设置成功
        """
        voice_path = os.path.join(self.voices_dir, f"{voice_name}.pt")
        if not os.path.exists(voice_path):
            logger.warning(f"声音文件不存在: {voice_path}")
            return False
            
        self.current_voice = voice_name
        self.config["default_voice"] = voice_name
        self._save_config()
        
        logger.info(f"已设置当前声音为: {voice_name}")
        return True
    
    def get_current_voice(self) -> str:
        """
        获取当前使用的声音
        
        Returns:
            str: 当前声音名称
        """
        return self.current_voice
    
    def add_custom_pronunciation(self, word: str, pronunciation: str) -> bool:
        """
        添加自定义发音规则
        
        Args:
            word: 单词
            pronunciation: 音标发音
            
        Returns:
            bool: 是否添加成功
        """
        if "custom_pronunciations" not in self.config:
            self.config["custom_pronunciations"] = {}
            
        self.config["custom_pronunciations"][word] = pronunciation
        success = self._save_config()
        
        # 更新英文发音回调
        if success and self.is_initialized:
            en_callable = self._create_en_callable()
            self.zh_pipeline.en_callable = en_callable
            logger.info(f"添加发音规则: {word} -> {pronunciation}")
            
        return success
    
    def remove_custom_pronunciation(self, word: str) -> bool:
        """
        删除自定义发音规则
        
        Args:
            word: 单词
            
        Returns:
            bool: 是否删除成功
        """
        if "custom_pronunciations" not in self.config or word not in self.config["custom_pronunciations"]:
            logger.warning(f"发音规则不存在: {word}")
            return False
            
        del self.config["custom_pronunciations"][word]
        success = self._save_config()
        
        # 更新英文发音回调
        if success and self.is_initialized:
            en_callable = self._create_en_callable()
            self.zh_pipeline.en_callable = en_callable
            logger.info(f"删除发音规则: {word}")
            
        return success
    
    def get_custom_pronunciations(self) -> Dict[str, str]:
        """
        获取所有自定义发音规则
        
        Returns:
            Dict[str, str]: 单词到发音的映射
        """
        return self.config.get("custom_pronunciations", {}).copy()
    
    def set_speed_factor(self, factor: float) -> bool:
        """
        设置全局语速因子
        
        Args:
            factor: 语速因子，1.0表示正常速度
            
        Returns:
            bool: 是否设置成功
        """
        if factor <= 0:
            logger.warning("语速因子必须大于0")
            return False
            
        self.config["speed_factor"] = factor
        success = self._save_config()
        
        if success:
            logger.info(f"设置语速因子: {factor}")
        
        return success

    async def shutdown(self) -> bool:
        """
        关闭并清理资源
        
        Returns:
            bool: 是否成功关闭
        """
        if not self.is_initialized:
            return True
            
        try:
            # 释放模型资源
            self.zh_model = None
            self.zh_pipeline = None
            self.en_pipeline = None
            
            # 触发垃圾回收
            import gc
            gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            self.is_initialized = False
            logger.info("已关闭语音合成系统")
            return True
        except Exception as e:
            logger.error(f"关闭语音合成系统失败: {e}")
            return False
