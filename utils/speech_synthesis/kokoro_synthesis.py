"""Kokoro语音合成模块，提供TTS功能."""
import gc  # Standard library imports first
import json
import logging
import os
import shutil
import time
from threading import Lock
from typing import Callable, Dict, List, Optional  # Added Optional
from dataclasses import dataclass, field  # Added dataclasses

import numpy as np  # Third party imports
import soundfile as sf
import torch
from huggingface_hub import snapshot_download as hf_snapshot_download
from kokoro import KModel, KPipeline

logging.basicConfig(
    level=logging.DEBUG if os.environ.get(
        'SYNTHESIS_DEBUG', '0') == '1' else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='./database/log/kokoro_synthesis.log',
    filemode='w',
    encoding='utf-8'
)
logger = logging.getLogger('KokoroSynthesis')

REPO_ID = 'hexgrad/Kokoro-82M-v1.1-zh'
# MODEL_NAME = 'kokoro-v1_1-zh.pth' # Unused if model path determined dynamically
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
                    cls._instances[cls] = super(
                        Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


@dataclass
class KokoroPaths:
    """
    Manages paths related to Kokoro model files and configurations.
    """
    model_dir_base: Optional[str] = None
    model_dir: str = field(init=False)
    voices_dir: str = field(init=False)
    model_cache_dir: str = field(init=False)
    config_path: str = field(init=False)

    def __post_init__(self):
        if self.model_dir_base is None:
            base_dir = os.path.dirname(
                os.path.dirname(os.path.abspath(__file__)))
            self.model_dir = os.path.join(base_dir, "models", "kokoro")
        else:
            self.model_dir = self.model_dir_base

        self.voices_dir = os.path.join(self.model_dir, "voices")
        self.model_cache_dir = os.path.join(self.model_dir, "model_cache")
        self.config_path = os.path.join(self.model_dir, "synthesis_config.json")

        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.voices_dir, exist_ok=True)
        os.makedirs(self.model_cache_dir, exist_ok=True)


@dataclass
class KokoroComponents:
    """
    Holds the initialized Kokoro model and pipeline components.
    """
    zh_model: Optional[KModel] = None
    zh_pipeline: Optional[KPipeline] = None
    en_pipeline: Optional[KPipeline] = None


class KokoroSynthesis(metaclass=Singleton):  # pylint: disable=too-many-instance-attributes
    """
    Kokoro语音合成单例类，提供流式TTS功能
    """

    def __init__(self, model_dir: str = None, config_path_override: str = None):
        """
        初始化Kokoro语音合成器

        Args:
            model_dir: 模型目录路径，默认为utils/models/kokoro
            config_path: 配置文件路径，默认在model_dir下
        """
        self.paths = KokoroPaths(model_dir_base=model_dir)
        if config_path_override:
            self.paths.config_path = config_path_override

        self.components = KokoroComponents()
        self.config = self._load_config()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.is_initialized = False
        self.current_voice = self.config.get("default_voice", DEFAULT_VOICE)

    def _load_config(self) -> dict:
        """加载配置文件，若不存在则创建默认配置"""
        if not os.path.exists(self.paths.config_path):
            logger.info("配置文件不存在，创建默认配置: %s", self.paths.config_path)
            default_config = DEFAULT_CONFIG.copy()
            default_config["voices_dir"] = self.paths.voices_dir  # Use path from KokoroPaths

            with open(self.paths.config_path, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, ensure_ascii=False, indent=4)
            return default_config

        try:
            with open(self.paths.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

                for key, value in DEFAULT_CONFIG.items():
                    if key not in config:
                        config[key] = value

                if not config.get("voices_dir"):  # Ensure voices_dir is correct
                    config["voices_dir"] = self.paths.voices_dir

            with open(self.paths.config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=4)

            return config
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("加载配置文件失败: %s，使用默认配置", e)
            default_config = DEFAULT_CONFIG.copy()
            default_config["voices_dir"] = self.paths.voices_dir
            return default_config

    def _save_config(self) -> bool:
        """保存当前配置到配置文件"""
        try:
            with open(self.paths.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=4)
            return True
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("保存配置文件失败: %s", e)
            return False

    def _check_existing_cache(
        self,
        model_id: str,
        model_cache_path: str,
        model_info_path: str
    ) -> Optional[str]:
        """Checks if a valid model exists in the local cache."""
        if os.path.exists(model_cache_path) and os.path.exists(model_info_path):
            try:
                with open(model_info_path, 'r', encoding='utf-8') as f:
                    model_info = json.load(f)
                logger.info("模型已存在本地: %s (%s)", model_id, model_info.get('timestamp'))
                return model_info.get('path', model_cache_path)
            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.warning("读取本地模型信息失败: %s", e)
                model_lock_path = os.path.join(model_cache_path, ".lock")
                if not os.path.exists(model_lock_path):
                    try:
                        logger.info("尝试清理损坏的模型缓存: %s", model_cache_path)
                        shutil.rmtree(model_cache_path, ignore_errors=True)
                        os.makedirs(model_cache_path, exist_ok=True)  # Recreate dir
                    except Exception as cleanup_err:  # pylint: disable=broad-exception-caught
                        logger.error("清理损坏的模型缓存失败: %s", cleanup_err)
        return None

    def _wait_for_other_download(
        self,
        model_lock_path: str,
        model_info_path: str,
        model_id: str,
        model_specific_cache_path: str  # Added parameter
    ) -> Optional[str]:
        """Waits if another process is downloading the model."""
        logger.warning("另一个进程正在下载模型，等待下载完成...")
        for _ in range(60):  # Wait for up to 10 minutes (60 * 10s)
            time.sleep(10)
            if not os.path.exists(model_lock_path):
                break  # Lock released
        if os.path.exists(model_lock_path):
            raise RuntimeError(
                "等待其他进程下载模型超时。"
                "如果确定没有其他下载正在进行，请手动删除锁文件: "
                f"{model_lock_path}"
            )
        # Check if model info appeared after waiting
        if os.path.exists(model_info_path):
            try:
                with open(model_info_path, 'r', encoding='utf-8') as f:
                    model_info = json.load(f)
                logger.info("模型已由另一进程下载完成: %s", model_id)
                return model_info.get('path', model_specific_cache_path)  # Use passed parameter
            except Exception:  # pylint: disable=broad-exception-caught
                pass  # Fall through to download
        return None

    def _perform_download(
        self,
        model_id: str,
        model_version: Optional[str],
        model_cache_path: str,
        model_info_path: str
    ) -> str:
        """Performs the actual model download from Hugging Face."""
        logger.info("正在从Hugging Face下载模型: %s %s", model_id, model_version or '')
        # Define local_dir within model_cache_path to keep everything contained
        local_model_dir = os.path.join(model_cache_path, "model_files")
        os.makedirs(local_model_dir, exist_ok=True)

        downloaded_path = hf_snapshot_download(
            repo_id=model_id,
            revision=model_version,
            cache_dir=self.paths.model_cache_dir,
            local_dir=local_model_dir,
            local_dir_use_symlinks=False
        )

        model_info = {
            "model_id": model_id,
            "version": model_version,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "path": downloaded_path  # This is the path to the directory containing model files
        }
        with open(model_info_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, ensure_ascii=False, indent=4)
        logger.info("模型下载成功: %s, 存储于: %s", model_id, downloaded_path)
        return downloaded_path

    def _download_and_cache_model(self, model_id: str, model_version: str = None) -> str:
        """
        Downloads and caches model, handling existing cache, locks, and download.
        """
        # Inlined cache_key to reduce local variables
        model_specific_cache_path = os.path.join(
            self.paths.model_cache_dir,
            f"{model_id.replace('/', '_')}" + (f"_{model_version}" if model_version else "")
        )
        model_info_path = os.path.join(model_specific_cache_path, "model_info.json")
        model_lock_path = os.path.join(model_specific_cache_path, ".lock")

        os.makedirs(model_specific_cache_path, exist_ok=True)

        cached_path = self._check_existing_cache(
            model_id, model_specific_cache_path, model_info_path)
        if cached_path:
            return cached_path

        if os.path.exists(model_lock_path):
            lock_modified_time = os.path.getmtime(model_lock_path)
            if time.time() - lock_modified_time > 1800:  # 30 minutes
                logger.warning("发现过期的锁文件，尝试移除: %s", model_lock_path)
                try:
                    os.remove(model_lock_path)
                except Exception:  # pylint: disable=broad-exception-caught
                    raise RuntimeError(
                        f"模型缓存目录被锁定，且无法释放锁。请手动删除: {model_lock_path}"
                    ) from None
            else:
                # Another process might be downloading, wait for it
                downloaded_by_other = self._wait_for_other_download(
                    model_lock_path,
                    model_info_path,
                    model_id,
                    model_specific_cache_path
                )
                if downloaded_by_other:
                    return downloaded_by_other

        # Acquire lock and download
        try:
            with open(model_lock_path, 'w', encoding='utf-8') as lock_file:  # Create/acquire lock
                lock_file.write(f"PID: {os.getpid()}, Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

            downloaded_model_path = self._perform_download(
                model_id, model_version, model_specific_cache_path, model_info_path)
            return downloaded_model_path

        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("模型下载或锁处理失败: %s", e, exc_info=True)
            # Mark as incomplete if download failed
            try:
                incomplete_marker = os.path.join(model_specific_cache_path, ".incomplete")
                with open(incomplete_marker, 'w', encoding='utf-8') as f_incomplete:
                    f_incomplete.write(f"Download failed: {str(e)}")
            except Exception:  # pylint: disable=broad-exception-caught
                pass
            raise RuntimeError(f"模型下载失败: {model_id}") from e
        finally:
            if os.path.exists(model_lock_path):
                try:
                    os.remove(model_lock_path)  # Release lock
                except Exception as del_err:  # pylint: disable=broad-exception-caught
                    logger.error("删除锁文件失败: %s", del_err)

    def _copy_voices_from_repo(self, repo_dir: str) -> bool:
        """从下载的模型仓库中复制声音文件到voices目录"""
        try:
            repo_voices_dir = os.path.join(repo_dir, "voices")
            if not (os.path.exists(repo_voices_dir) and os.path.isdir(repo_voices_dir)):
                logger.warning("仓库中不存在声音目录: %s", repo_voices_dir)
                return False

            voice_files = [f for f in os.listdir(repo_voices_dir) if f.endswith('.pt')]
            if not voice_files:
                logger.warning("仓库中未找到声音文件: %s", repo_voices_dir)
                return False

            for voice_file in voice_files:
                src_path = os.path.join(repo_voices_dir, voice_file)
                dst_path = os.path.join(self.paths.voices_dir, voice_file)
                if not os.path.exists(dst_path):
                    shutil.copy2(src_path, dst_path)
                    logger.info("复制声音文件: %s", voice_file)
            return True
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("复制声音文件失败: %s", e)
            return False

    def _create_en_callable(self) -> Callable:
        """创建英文发音回调函数"""
        custom_pronunciations = self.config.get("custom_pronunciations", {})

        def en_callable(text):
            if text in custom_pronunciations:
                return custom_pronunciations[text]
            return next(self.components.en_pipeline(text)).phonemes # Corrected attribute access

        return en_callable

    def _speed_callable(self, len_ps) -> float:
        """
        创建语速调整回调函数
        对于较长的文本，减慢语速以提高质量
        """
        base_speed = self.config.get("speed_factor", 1.0)

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
            logger.info("正在初始化Kokoro语音合成模型，使用设备: %s", self.device)
            downloaded_model_base_path = self._download_and_cache_model(REPO_ID)

            model_files = [f for f in os.listdir(downloaded_model_base_path) if f.endswith('.pth')]
            config_files = [f for f in os.listdir(downloaded_model_base_path) if f == CONFIG_NAME]

            if not model_files:
                raise FileNotFoundError(f"在下载的模型目录中未找到模型文件: {downloaded_model_base_path}")
            if not config_files:
                raise FileNotFoundError(f"在下载的模型目录中未找到配置文件: {downloaded_model_base_path}")

            model_file_path = os.path.join(downloaded_model_base_path, model_files[0])
            config_file_path = os.path.join(downloaded_model_base_path, config_files[0])

            self.config["model_path"] = model_file_path # Save the actual .pth path
            self._save_config()
            self._copy_voices_from_repo(downloaded_model_base_path)

            self.components.zh_model = KModel(
                repo_id=REPO_ID, # Still useful for KModel internals if it uses it
                config=config_file_path,
                model=model_file_path
            ).to(self.device).eval()
            logger.info("Kokoro语音模型加载成功")

            self.components.en_pipeline = KPipeline(lang_code='a', repo_id=REPO_ID, model=False)
            en_callable = self._create_en_callable()

            self.components.zh_pipeline = KPipeline(
                lang_code='z',
                repo_id=REPO_ID,
                model=self.components.zh_model,
                en_callable=en_callable
            )
            self._ensure_voices_exist()
            self.is_initialized = True
            logger.info("Kokoro语音合成系统初始化完成")
            return True

        except Exception as e:  # pylint: disable=broad-exception-caught
            # Fallback to local path if primary download/load fails and a model_path is in config
            logger.error("模型初始化/下载失败: %s. 尝试本地路径...", e)
            if self.config.get("model_path") and os.path.exists(self.config["model_path"]):
                try:
                    logger.info("尝试从本地路径加载模型: %s", self.config['model_path'])
                    local_model_file_path = self.config["model_path"]
                    local_config_file_path = os.path.join(
                        os.path.dirname(local_model_file_path), CONFIG_NAME)

                    if not os.path.exists(local_config_file_path):
                        raise FileNotFoundError(
                            f"本地配置文件未找到: {local_config_file_path}"
                        ) from e


                    self.components.zh_model = KModel(
                        config=local_config_file_path,
                        model=local_model_file_path
                    ).to(self.device).eval()
                    logger.info("从本地路径加载Kokoro语音模型成功")

                    # Re-initialize pipelines with the locally loaded model
                    self.components.en_pipeline = KPipeline(
                        lang_code='a', repo_id=REPO_ID, model=False)
                    en_callable = self._create_en_callable()
                    self.components.zh_pipeline = KPipeline(
                        lang_code='z', model=self.components.zh_model, en_callable=en_callable
                    )
                    self._ensure_voices_exist()
                    self.is_initialized = True
                    logger.info("Kokoro语音合成系统从本地路径初始化完成")
                    return True
                except Exception as local_load_e: # pylint: disable=broad-exception-caught
                    logger.error("从本地路径加载模型也失败: %s", local_load_e)

            self.is_initialized = False # Ensure it's marked as not initialized
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

                    voices = self.get_available_voices()
                    if not voices:
                        logger.error("声音文件下载失败")
                        return False
                    # Removed unnecessary else, de-indented the following return
                    return True
                # else: # This else is removed
                logger.error("模型下载失败，无法提取声音文件")
                return False
            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.error("声音文件获取失败: %s", e)
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

        voice_name = voice or self.current_voice
        if voice_name != self.current_voice:  # This check is fine
            if not self.set_current_voice(voice_name): # Ensure voice exists before using
                 # If set_current_voice failed (e.g. voice not found), use a fallback
                available_voices = self.get_available_voices()
                if not available_voices:
                    raise FileNotFoundError("未找到可用的声音文件，且设置默认声音失败")
                voice_name = available_voices[0]
                self.set_current_voice(voice_name) # Try setting to the first available
                logger.warning("指定的声音 %s 不存在, 使用可用的第一个声音: %s",
                               voice or self.current_voice, voice_name)


        voice_path = os.path.join(self.paths.voices_dir, f"{voice_name}.pt")
        if not os.path.exists(voice_path): # Double check, though set_current_voice should handle
            available_voices = self.get_available_voices()
            if not available_voices:
                raise FileNotFoundError("未找到可用的声音文件")
            voice_name = available_voices[0] # Fallback to first available
            voice_path = os.path.join(self.paths.voices_dir, f"{voice_name}.pt")
            logger.warning(
                "指定的声音文件 %s 不存在，回退到可用的第一个声音: %s",
                os.path.join(self.paths.voices_dir, f"{voice or self.current_voice}.pt"),
                voice_name
            )
            self.current_voice = voice_name # Update current_voice if fallback occurred here
            self.config["default_voice"] = voice_name
            self._save_config()


        speed_callable_fn = None
        if speed is not None:
            def s_fn(_):
                return speed
            speed_callable_fn = s_fn
        else:
            speed_callable_fn = self._speed_callable

        try:
            logger.debug("开始合成文本: '%s', 使用声音: %s", text, voice_name)
            if not self.components.zh_pipeline:
                raise RuntimeError("中文语音合成管道未初始化。")
            generator = self.components.zh_pipeline(
                text, voice=voice_path, speed=speed_callable_fn)
            result = next(generator)
            logger.debug("语音合成完成")
            return result.audio
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("语音合成失败: %s", e)
            raise

    async def synthesize_to_file(
            self,
            text: str,
            output_file: str,
            voice: str = None,
            speed: float = None
    ) -> str:
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
            output_dir = os.path.dirname(output_file)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            sf.write(output_file, audio_data, SAMPLE_RATE)
            logger.info("语音已保存到文件: %s", output_file)
            return output_file
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("保存语音文件失败: %s", e)
            raise

    def get_available_voices(self) -> List[str]:
        """
        获取可用的声音列表

        Returns:
            List[str]: 可用声音名称列表
        """
        voices = []
        try:
            if not os.path.exists(self.paths.voices_dir):
                os.makedirs(self.paths.voices_dir, exist_ok=True)

            voice_files = [f for f in os.listdir(
                self.paths.voices_dir) if f.endswith('.pt')]
            voices = [os.path.splitext(f)[0] for f in voice_files]
            return voices
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("获取可用声音列表失败: %s", e)
            return []

    def set_current_voice(self, voice_name: str) -> bool:
        """
        设置当前使用的声音
        """
        voice_path = os.path.join(self.paths.voices_dir, f"{voice_name}.pt")
        if not os.path.exists(voice_path):
            logger.warning("声音文件不存在: %s", voice_path)
            return False

        self.current_voice = voice_name
        self.config["default_voice"] = voice_name
        self._save_config()

        logger.info("已设置当前声音为: %s", voice_name)
        return True

    def get_current_voice(self) -> str:  # Ensure this is the only definition
        """
        获取当前使用的声音
        """
        return self.current_voice
    # Removed any duplicate get_current_voice method and fixed multi-statement lines if any existed.

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

        if success and self.is_initialized and self.components.zh_pipeline:
            en_callable = self._create_en_callable()
            self.components.zh_pipeline.en_callable = en_callable
            logger.info("添加发音规则: %s -> %s", word, pronunciation)

        return success

    def remove_custom_pronunciation(self, word: str) -> bool:
        """
        删除自定义发音规则

        Args:
            word: 单词

        Returns:
            bool: 是否删除成功
        """
        if "custom_pronunciations" not in self.config \
            or word not in self.config["custom_pronunciations"]:
            logger.warning("发音规则不存在: %s", word)
            return False

        del self.config["custom_pronunciations"][word]
        success = self._save_config()

        if success and self.is_initialized and self.components.zh_pipeline:
            en_callable = self._create_en_callable()
            self.components.zh_pipeline.en_callable = en_callable
            logger.info("删除发音规则: %s", word)

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
            logger.info("设置语速因子: %s", factor)

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
            self.components.zh_model = None # Use components
            self.components.zh_pipeline = None
            self.components.en_pipeline = None

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.is_initialized = False
            logger.info("已关闭语音合成系统")
            return True
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("关闭语音合成系统失败: %s", e)
            return False
