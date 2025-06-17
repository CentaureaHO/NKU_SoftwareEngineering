"""语音识别模块，提供智能座舱语音识别功能"""

import logging
import os
import sys
import threading
import time
from queue import Queue
from typing import Dict, List, Optional, Any

import numpy as np
import pyaudio
import webrtcvad
from funasr import AutoModel
from modelscope.pipelines import pipeline

from modality.core.base_modality import BaseModality
from modality.core.error_codes import (ALREADY_INITIALIZED,
                                       MODEL_LOADING_FAILED, NOT_INITIALIZED,
                                       OPERATION_FAILED, SUCCESS)
from modality.speech.config_manager import ConfigManager
from modality.speech.model_manager import ModelManager
from modality.speech.speaker_manager import SpeakerManager
from modality.speech.speech_processor import SpeechProcessor, AudioConfig
from modality.speech.speech_state import SpeechState

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.DEBUG if os.environ.get(
        'MODALITY_DEBUG', '0') == '1' else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='./database/log/speech_recognition.log',
    filemode='w',
    encoding='utf-8'
)
logger = logging.getLogger('SpeechRecognition')

DEBUG = os.environ.get('MODALITY_DEBUG', '0') == '1'

# 模型配置
ASR_MODEL_ID = "iic/SenseVoiceSmall"
SV_MODEL_ID = "damo/speech_campplus_sv_zh-cn_16k-common"
SV_MODEL_VERSION = "v1.0.0"

# 音频配置
AUDIO_CONFIG = {
    "rate": 16000,          # 采样率
    "chunk": 1024,          # 音频数据块大小
    "channels": 1,          # 单声道
    "format": pyaudio.paInt16,  # 16位PCM格式
    "vad_mode": 3,          # VAD敏感度 (0-3, 越高越敏感)
    "silence_threshold": 1.0  # 无声间隔阈值(秒)
}


class SpeechRecognition(BaseModality):
    """智能座舱语音识别模块，继承自BaseModality"""

    def __init__(self, name: str = "speech_recognition",
                 config_path: str = None,
                 model_dir: str = None,
                 debug: bool = DEBUG):
        """
        初始化语音识别模态

        Args:
            name: 模态名称
            config_path: 配置文件路径，若为None则使用默认配置
            model_dir: 模型目录路径
            debug: 是否启用调试模式
        """
        super().__init__(name)

        self._context = {
            "debug": debug,
            "resources": self._init_resources(model_dir, config_path),
            "processor": None  # 将在initialize()中创建
        }

    def _init_resources(self, model_dir: Optional[str], config_path: Optional[str]) -> Dict:
        """初始化资源和组件"""
        dirs = self._init_directories(model_dir, config_path)

        config_manager = ConfigManager(dirs["config_path"])
        config = config_manager.get_config()
        model_manager = ModelManager(dirs["model_cache_dir"])
        speaker_manager = SpeakerManager(
            dirs["speaker_db_dir"],
            dirs["temp_speaker_dir"],
            config.get("max_temp_speakers", 10)
        )

        resources = {
            "dirs": dirs,
            "managers": {
                "config": config_manager,
                "model": model_manager,
                "speaker": speaker_manager
            },
            "config": config,
            "models": {
                "vad": None,
                "asr_model": None,
                "sv_model": None,
                "speakers": speaker_manager.get_speakers(),
                "temp_speakers": speaker_manager.get_temp_speakers()
            },
            "state": {
                "is_recording": False,
                "is_listening": not config["enable_wake_word"],
                "is_enrolling": False,
                "current_speaker_id": None,
                "enrolling_name": None,
                "last_key_info": None
            },
            "audio": {
                "queue": Queue(),
                "segments": [],
                "last_voice_time": time.time()
            },
            "threads": {
                "record": None,
                "process": None,
                "stop_event": threading.Event(),
                "state_lock": threading.Lock(),
                "latest_state": SpeechState()
            }
        }

        return resources

    def _init_directories(
        self,
        model_dir: Optional[str],
        config_path: Optional[str]
    ) -> Dict[str, str]:
        """初始化相关目录，返回包含路径的字典"""
        if model_dir is None:
            base_dir = os.path.dirname(
                os.path.dirname(os.path.abspath(__file__)))
            model_dir = os.path.join(base_dir, "models", "speech")

        dirs = {
            "model_dir": model_dir,
            "output_dir": os.path.join(model_dir, "output"),
            "speaker_db_dir": os.path.join(model_dir, "speaker_db"),
            "temp_speaker_dir": os.path.join(model_dir, "temp_speakers"),
            "model_cache_dir": os.path.join(model_dir, "model_cache"),
            "config_path": os.path.join(model_dir, "config.json") if config_path is None
            else config_path
        }

        for key, path in dirs.items():
            if key.endswith("_dir"):
                os.makedirs(path, exist_ok=True)

        return dirs

    @property
    def is_recording(self) -> bool:
        """是否正在录音"""
        return self._context["resources"]["state"]["is_recording"]

    @is_recording.setter
    def is_recording(self, value: bool):
        self._context["resources"]["state"]["is_recording"] = value

    @property
    def is_listening(self) -> bool:
        """是否处于监听状态"""
        return self._context["resources"]["state"]["is_listening"]

    @is_listening.setter
    def is_listening(self, value: bool):
        self._context["resources"]["state"]["is_listening"] = value

    @property
    def is_enrolling(self) -> bool:
        """是否处于声纹注册状态"""
        return self._context["resources"]["state"]["is_enrolling"]

    @is_enrolling.setter
    def is_enrolling(self, value: bool):
        self._context["resources"]["state"]["is_enrolling"] = value

    @property
    def config(self) -> Dict:
        """获取配置"""
        return self._context["resources"]["config"]

    @config.setter
    def config(self, value: Dict):
        self._context["resources"]["config"] = value

    @property
    def audio_segments(self) -> List:
        """获取音频段列表"""
        return self._context["resources"]["audio"]["segments"]

    @audio_segments.setter
    def audio_segments(self, value: List):
        self._context["resources"]["audio"]["segments"] = value

    @property
    def _resources(self) -> Dict:
        """获取资源字典"""
        return self._context["resources"]

    @property
    def _processor(self) -> Optional[SpeechProcessor]:
        """获取语音处理器"""
        return self._context["processor"]

    @_processor.setter
    def _processor(self, value: SpeechProcessor):
        self._context["processor"] = value

    def initialize(self) -> int:
        """
        初始化语音识别模块

        Returns:
            int: 错误码，0表示成功，其他值表示失败
        """
        if self._is_running:
            return ALREADY_INITIALIZED

        try:
            # 初始化VAD
            vad = webrtcvad.Vad()
            vad.set_mode(AUDIO_CONFIG["vad_mode"])
            self._resources["models"]["vad"] = vad
            logger.info("VAD初始化成功")

            logger.info("正在初始化语音识别模型...")

            # 使用模型管理器加载ASR模型
            try:
                asr_model_path = self._resources["managers"]["model"].download_and_cache_model(
                    ASR_MODEL_ID)
                self._resources["models"]["asr_model"] = AutoModel(
                    model=asr_model_path, trust_remote_code=True)
                logger.info("语音识别模型加载成功")
            except (RuntimeError, FileNotFoundError, ImportError, OSError) as e:
                logger.error("语音识别模型加载失败: %s", e)
                return MODEL_LOADING_FAILED

            # 加载声纹识别模型（如果启用）
            if self.config["enable_speaker_verification"]:
                logger.info("正在初始化声纹识别模型...")
                try:
                    sv_model_path = self._resources["managers"]["model"].download_and_cache_model(
                        SV_MODEL_ID, SV_MODEL_VERSION)
                    self._resources["models"]["sv_model"] = pipeline(
                        task='speaker-verification',
                        model=SV_MODEL_ID,
                        model_revision=SV_MODEL_VERSION,
                        cache_dir=self._resources["dirs"]["model_cache_dir"]
                    )
                    logger.info("声纹识别模型加载成功，路径: %s", sv_model_path)
                except (RuntimeError, FileNotFoundError, ImportError, OSError) as e:
                    logger.error("声纹识别模型加载失败: %s", e)
                    return MODEL_LOADING_FAILED

            # 更新说话人信息
            self._resources["models"]["speakers"] =\
                self._resources["managers"]["speaker"].get_speakers()
            self._resources["models"]["temp_speakers"] =\
                self._resources["managers"]["speaker"].get_temp_speakers()

            # 清空音频队列
            with self._resources["audio"]["queue"].mutex:
                self._resources["audio"]["queue"].queue.clear()
            self.audio_segments = []

            # 重置状态
            self._resources["state"]["is_recording"] = False
            self._resources["state"]["is_listening"] = not self.config["enable_wake_word"]
            self._resources["state"]["is_enrolling"] = False
            self._resources["state"]["current_speaker_id"] = None

            # 初始化语音处理器
            self._processor = SpeechProcessor(
                self.config,
                self._resources["dirs"],
                self._resources["models"]
            )

            logger.info("语音识别模块初始化完成")
            logger.info(
                "唤醒词功能: %s", '开启' if self.config['enable_wake_word'] else '关闭')
            logger.info(
                "声纹识别功能: %s", '开启' if self.config['enable_speaker_verification'] else '关闭')
            logger.info("已注册说话人: %d个", len(
                self._resources["models"]["speakers"]))
            logger.info("临时声纹数量: %d个", len(
                self._resources["models"]["temp_speakers"]))

            return SUCCESS
        except (RuntimeError, ImportError, OSError) as e:
            logger.error("初始化失败: %s", e)
            return MODEL_LOADING_FAILED

    def process_audio_chunk(self, audio_chunk: bytes):
        """处理音频块"""
        if not self.is_recording or not self._is_running or self._processor is None:
            return

        audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
        vad_active = self._processor.check_vad_activity(
            audio_data, AUDIO_CONFIG["rate"])

        if vad_active:
            self.audio_segments.append(audio_chunk)
            self._resources["audio"]["last_voice_time"] = time.time()

            if not self.is_listening and\
                self.config["enable_wake_word"] and\
                    len(self.audio_segments) == 1:
                logger.debug("检测到声音，等待唤醒词...")
        else:
            if self.audio_segments and\
                (time.time() - self._resources["audio"]["last_voice_time"])\
                    > AUDIO_CONFIG["silence_threshold"]:
                self._process_audio_segments()

    def _process_wake_word_detection(self, temp_file: str) -> Optional[SpeechState]:
        """非监听状态下检测唤醒词"""
        text = self._processor.recognize_speech(temp_file)
        if not text:
            return None

        state = SpeechState()
        state.recognition["text"] = text

        wake_word = self._processor.detect_wake_word(text)
        if not wake_word:
            return None

        logger.info("检测到唤醒词: %s", wake_word)
        self.is_listening = True

        state.recognition["wake_word"] = wake_word
        state.recognition["has_wake_word"] = True
        state.recognition["is_command"] = True

        # 识别说话人
        if self.config["enable_speaker_verification"]:
            speaker_match = self._processor.identify_speaker(temp_file)
            if speaker_match.speaker_id:
                self._resources["state"]["current_speaker_id"] = speaker_match.speaker_id
                speaker_info = self._resources["models"]["speakers"].get(speaker_match.speaker_id)\
                    if speaker_match.is_registered else\
                    self._resources["models"]["temp_speakers"].get(
                        speaker_match.speaker_id)
                speaker_name = speaker_info["name"] if speaker_info\
                    else f"用户{speaker_match.speaker_id}"

                state.recognition["speaker_id"] = speaker_match.speaker_id
                state.recognition["speaker_name"] = speaker_name
                state.recognition["is_registered_speaker"] = speaker_match.is_registered
                state.recognition["confidence"] = speaker_match.similarity

                logger.info("识别到说话人: %s (%s)",
                            speaker_name, "已注册" if speaker_match.is_registered else "临时")

        return state

    def _process_listening_mode(self, temp_file: str) -> SpeechState:
        """处理监听状态下的音频"""
        state = SpeechState()

        # 识别语音内容
        text = self._processor.recognize_speech(temp_file)
        state.recognition["text"] = text

        if not text:
            return state

        # 检查唤醒词
        wake_word = self._processor.detect_wake_word(text)
        if wake_word:
            state.recognition["wake_word"] = wake_word
            state.recognition["has_wake_word"] = True
            state.recognition["is_command"] = True

        # 检查关键词
        keyword, category = self._processor.detect_keyword(text)
        if keyword:
            logger.info("检测到关键词: %s (%s)", keyword, category)
            state.recognition["keyword"] = keyword
            state.recognition["keyword_category"] = category
            state.recognition["has_keyword"] = True
            state.recognition["is_command"] = True

        # 识别说话人
        if self.config["enable_speaker_verification"]:
            speaker_match = self._processor.identify_speaker(temp_file)

            if speaker_match.speaker_id:
                self._resources["state"]["current_speaker_id"] = speaker_match.speaker_id
                speaker_info = self._resources["models"]["speakers"].get(speaker_match.speaker_id)\
                    if speaker_match.is_registered else\
                    self._resources["models"]["temp_speakers"].get(
                        speaker_match.speaker_id)
                speaker_name = speaker_info["name"] if speaker_info else f"用户{
                    speaker_match.speaker_id}"

                state.recognition["speaker_id"] = speaker_match.speaker_id
                state.recognition["speaker_name"] = speaker_name
                state.recognition["is_registered_speaker"] = speaker_match.is_registered
                state.recognition["confidence"] = speaker_match.similarity

                logger.info("识别到说话人: %s (%s)",
                            speaker_name, "已注册" if speaker_match.is_registered else "临时")
            else:
                # 保存未识别的临时声纹
                self._resources["managers"]["speaker"].save_temp_speaker(
                    temp_file)
                self._resources["models"]["temp_speakers"] =\
                    self._resources["managers"]["speaker"].get_temp_speakers()

        # 检查是否有退出监听的关键词
        if self.config["enable_wake_word"] and self.is_listening:
            exit_keywords = self._resources["managers"]["config"].get_exit_keywords(
            )
            if any(exit_word in text for exit_word in exit_keywords):
                logger.info("检测到结束指令： %s", text)
                self.is_listening = False

        return state

    def _process_audio_segments(self):
        """处理收集到的音频段落"""
        if not self.audio_segments or self._processor is None:
            return

        audio_data = b''.join(self.audio_segments)
        self.audio_segments = []

        # 声纹注册模式
        if self.is_enrolling:
            enrolling_config = {
                "name": self._resources["state"]["enrolling_name"],
                "audio_config": AudioConfig(
                    rate=AUDIO_CONFIG["rate"],
                    channels=AUDIO_CONFIG["channels"]
                ),
                "speaker_manager": self._resources["managers"]["speaker"]
            }

            state = self._processor.process_enrollment(
                audio_data, enrolling_config)

            if state:
                self.is_enrolling = False
                self._resources["models"]["speakers"] =\
                    self._resources["managers"]["speaker"].get_speakers()

                with self._resources["threads"]["state_lock"]:
                    self._resources["threads"]["latest_state"] = state
            return

        # 保存临时音频文件用于处理
        temp_file = os.path.join(
            self._resources["dirs"]["output_dir"], "temp_audio.wav")
        self._processor.save_audio(
            audio_data, temp_file,
            AUDIO_CONFIG["rate"], AUDIO_CONFIG["channels"]
        )

        # 非监听状态下检测唤醒词
        if not self.is_listening and self.config["enable_wake_word"]:
            state = self._process_wake_word_detection(temp_file)
            if state:
                with self._resources["threads"]["state_lock"]:
                    self._resources["threads"]["latest_state"] = state
            return

        # 监听状态下的处理
        state = self._process_listening_mode(temp_file)

        # 更新状态
        with self._resources["threads"]["state_lock"]:
            self._resources["threads"]["latest_state"] = state

    def _audio_recording_thread(self):
        """音频录制线程"""
        logger.info("音频录制线程启动")

        p = pyaudio.PyAudio()

        try:
            stream = p.open(format=AUDIO_CONFIG["format"],
                            channels=AUDIO_CONFIG["channels"],
                            rate=AUDIO_CONFIG["rate"],
                            input=True,
                            frames_per_buffer=AUDIO_CONFIG["chunk"])

            logger.info("录音开始")

            while not self._resources["threads"]["stop_event"].is_set() and self.is_recording:
                if self._resources["audio"]["queue"].qsize() < 100:
                    data = stream.read(
                        AUDIO_CONFIG["chunk"], exception_on_overflow=False)
                    self._resources["audio"]["queue"].put(data)
                else:
                    time.sleep(0.01)
        except (IOError, OSError) as e:
            logger.error("录音线程错误: %s", e)
        finally:
            if 'stream' in locals():
                stream.stop_stream()
                stream.close()
            p.terminate()
            logger.info("录音结束")

    def _audio_processing_thread(self):
        """音频处理线程"""
        logger.info("音频处理线程启动")

        try:
            while not self._resources["threads"]["stop_event"].is_set():
                if not self._resources["audio"]["queue"].empty():
                    audio_data = self._resources["audio"]["queue"].get()
                    self.process_audio_chunk(audio_data)
                else:
                    time.sleep(0.01)
        except (RuntimeError, IOError) as e:
            logger.error("音频处理线程错误: %s", e)
        finally:
            logger.info("音频处理线程结束")

    def start(self) -> int:
        """
        启动语音识别系统

        Returns:
            int: 错误码，0表示成功，其他值表示失败
        """
        result = super().start()
        if result != SUCCESS:
            logger.error("无法启动语音识别系统: %s", result)
            return result

        try:
            self._resources["threads"]["stop_event"].clear()
            self.is_recording = True

            self._resources["threads"]["record"] = threading.Thread(
                target=self._audio_recording_thread)
            self._resources["threads"]["record"].daemon = True
            self._resources["threads"]["record"].start()

            self._resources["threads"]["process"] = threading.Thread(
                target=self._audio_processing_thread)
            self._resources["threads"]["process"].daemon = True
            self._resources["threads"]["process"].start()

            logger.info("语音识别系统启动成功")
            if self.config["enable_wake_word"]:
                logger.info("请说唤醒词以激活语音识别系统")
            else:
                logger.info("已进入语音识别模式")

            return SUCCESS
        except (RuntimeError, threading.ThreadError) as e:
            logger.error("启动处理线程时出错: %s", e)
            return OPERATION_FAILED

    def shutdown(self) -> int:
        """
        关闭语音识别系统

        Returns:
            int: 错误码，0表示成功，其他值表示失败
        """
        if not self._is_running:
            return NOT_INITIALIZED

        try:
            logger.info("正在关闭语音识别系统...")

            self._resources["threads"]["stop_event"].set()
            self.is_recording = False

            record_thread = self._resources["threads"]["record"]
            process_thread = self._resources["threads"]["process"]

            if record_thread and record_thread.is_alive():
                record_thread.join(timeout=2.0)
            if process_thread and process_thread.is_alive():
                process_thread.join(timeout=2.0)

            self._resources["threads"]["record"] = None
            self._resources["threads"]["process"] = None

            self._resources["models"]["asr_model"] = None
            self._resources["models"]["sv_model"] = None
            self._resources["models"]["vad"] = None

            with self._resources["audio"]["queue"].mutex:
                self._resources["audio"]["queue"].queue.clear()
            self.audio_segments = []

            logger.info("语音识别系统已关闭")
            return SUCCESS
        except (RuntimeError, OSError) as e:
            logger.error("关闭语音识别系统失败: %s", e)
            return OPERATION_FAILED

    def update(self) -> Optional[SpeechState]:
        """
        获取最新的语音识别状态

        Returns:
            Optional[SpeechState]: 当前语音识别状态或None
        """
        if not self._is_running:
            return None

        with self._resources["threads"]["state_lock"]:
            return self._resources["threads"]["latest_state"]

    # ===== 公共API方法 =====

    def manage_wake_word(
        self,
        action: str,
        wake_word: Optional[str] = None,
        enable: Optional[bool] = None
    ) -> Any:
        """
        管理唤醒词（统一多个唤醒词相关操作）

        Args:
            action: 操作类型，可选值: "add", "remove", "toggle", "get"
            wake_word: 要操作的唤醒词
            enable: 启用状态，仅当action为"toggle"时有效

        Returns:
            Any: 根据操作类型返回不同结果
        """
        config_manager = self._resources["managers"]["config"]
        result = False

        if action == "add" and wake_word:
            result = config_manager.add_wake_word(wake_word)
        elif action == "remove" and wake_word:
            result = config_manager.remove_wake_word(wake_word)
        elif action == "toggle":
            result = config_manager.toggle_wake_word(enable)
            self.is_listening = not self.config["enable_wake_word"]
        elif action == "get":
            return self.config.get("wake_words", [])
        else:
            logger.warning("无效的唤醒词管理操作: %s", action)
            return False

        if result and action != "get":
            self.config = config_manager.get_config()

        return result

    def manage_keywords(
        self,
        action: str,
        keyword: Optional[str] = None,
        category: Optional[str] = None
    ) -> Any:
        """
        管理关键词（统一多个关键词相关操作）

        Args:
            action: 操作类型，可选值: "add", "remove", "get"
            keyword: 要操作的关键词
            category: 关键词类别，仅当action为"add"时必须

        Returns:
            Any: 根据操作类型返回不同结果
        """
        config_manager = self._resources["managers"]["config"]
        result = False

        if action == "add" and keyword and category:
            result = config_manager.add_keyword(keyword, category)
        elif action == "remove" and keyword:
            result = config_manager.remove_keyword(keyword)
        elif action == "get":
            return self.config.get("keywords", {})
        else:
            logger.warning("无效的关键词管理操作: %s", action)
            return False

        # 如果操作成功，更新配置
        if result and action != "get":
            self.config = config_manager.get_config()

        return result

    def manage_exit_keywords(self, action: str, keyword: Optional[str] = None) -> Any:
        """
        管理退出关键词（统一多个退出关键词相关操作）

        Args:
            action: 操作类型，可选值: "add", "remove", "get"
            keyword: 要操作的关键词

        Returns:
            Any: 根据操作类型返回不同结果
        """
        config_manager = self._resources["managers"]["config"]
        result = False

        if action == "add" and keyword:
            result = config_manager.add_exit_keyword(keyword)
        elif action == "remove" and keyword:
            result = config_manager.remove_exit_keyword(keyword)
        elif action == "get":
            return config_manager.get_exit_keywords()
        else:
            logger.warning("无效的退出关键词管理操作: %s", action)
            return False

        # 如果操作成功，更新配置
        if result and action != "get":
            self.config = config_manager.get_config()

        return result

    def manage_speakers(
        self,
        action: str,
        speaker_id: Optional[str] = None,
        name: Optional[str] = None
    ) -> Any:
        """
        管理说话人（统一多个说话人相关操作）

        Args:
            action: 操作类型，可选值: "register", "delete", "delete_temp", "promote", "get", "get_temp"
            speaker_id: 说话人或临时说话人ID
            name: 说话人名称

        Returns:
            Any: 根据操作类型返回不同结果
        """
        speaker_manager = self._resources["managers"]["speaker"]

        # 注册新说话人
        if action == "register":
            if not self.config["enable_speaker_verification"]:
                logger.warning("声纹识别功能未开启")
                return False

            print("进入声纹注册模式，请说话...")
            logger.info("进入声纹注册模式，请说话...")
            self.is_enrolling = True
            self.audio_segments = []

            # 生成新的说话人ID并存储名称
            speaker_id = speaker_manager.generate_speaker_id()
            self._resources["state"]["enrolling_name"] = name if name else f"用户{
                speaker_id}"
            return True

        # 处理其它说话人管理操作
        result = False

        if action == "delete" and speaker_id:
            result = speaker_manager.delete_speaker(speaker_id)
            if result:
                self._resources["models"]["speakers"] = speaker_manager.get_speakers(
                )
        elif action == "delete_temp" and speaker_id:
            result = speaker_manager.delete_temp_speaker(speaker_id)
            if result:
                self._resources["models"]["temp_speakers"] = speaker_manager.get_temp_speakers(
                )
        elif action == "promote" and speaker_id:
            result = speaker_manager.promote_temp_speaker(speaker_id, name)
            if result:
                self._resources["models"]["speakers"] = speaker_manager.get_speakers(
                )
                self._resources["models"]["temp_speakers"] = speaker_manager.get_temp_speakers(
                )
        elif action == "get":
            return speaker_manager.get_speakers()
        elif action == "get_temp":
            return speaker_manager.get_temp_speakers()
        else:
            logger.warning("无效的说话人管理操作: %s", action)

        return result

    def set_max_temp_speakers(self, count: int) -> bool:
        """
        设置最大临时声纹数量

        Args:
            count: 最大临时声纹数量

        Returns:
            bool: 是否设置成功
        """
        if count < 1:
            logger.warning("最大临时声纹数量不能小于1")
            return False

        result = self._resources["managers"]["speaker"].set_max_temp_speakers(
            count)
        if result:
            # 更新配置
            self.config["max_temp_speakers"] = count
            self._resources["managers"]["config"].update_config(
                {"max_temp_speakers": count})
        return result

    def toggle_listening(self, enable: Optional[bool] = None) -> bool:
        """
        切换聆听状态

        Args:
            enable: 是否启用聆听，None表示切换当前状态

        Returns:
            bool: 操作后的聆听状态
        """
        if enable is not None:
            if enable == self.is_listening:
                logger.warning("已处于%s状态", "聆听" if enable else "非聆听")
                return self.is_listening

            self.is_listening = enable
        else:
            self.is_listening = not self.is_listening

        logger.info("已%s聆听状态", "启用" if self.is_listening else "禁用")
        return self.is_listening

    def get_key_info(self) -> Optional[str]:
        """
        获取模态的关键信息

        Returns:
            Optional[str]: 模态的关键信息，若无则返回None
        """
        key_info = None
        state = self.update()
        if state and state.recognition["text"]:
            key_info = state.recognition["text"]
            state.recognition["text"] = ""
        return key_info
