import os
import time
import wave
import json
import uuid
import numpy as np
import threading
import pyaudio
import webrtcvad
from queue import Queue
from pypinyin import lazy_pinyin
from typing import Dict, Any, List, Optional
import logging
import shutil
from modelscope.pipelines import pipeline
from modelscope.hub.snapshot_download import snapshot_download
from funasr import AutoModel

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modality.core.base_modality import BaseModality, ModalityState
from modality.core.error_codes import (
    SUCCESS, ALREADY_INITIALIZED, NOT_INITIALIZED, 
    OPERATION_FAILED, MODEL_LOADING_FAILED
)

logging.basicConfig(
    level=logging.DEBUG if os.environ.get('MODALITY_DEBUG', '0') == '1' else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='speech_recognition.log',
    filemode='w'
)
logger = logging.getLogger('SpeechRecognition')

DEBUG = os.environ.get('MODALITY_DEBUG', '0') == '1'

ASR_MODEL_ID = "iic/SenseVoiceSmall"
SV_MODEL_ID = "damo/speech_campplus_sv_zh-cn_16k-common"
SV_MODEL_VERSION = "v1.0.0"

RATE = 16000                # 采样率
CHUNK = 1024                # 音频数据块大小
CHANNELS = 1                # 单声道
FORMAT = pyaudio.paInt16    # 16位PCM格式
VAD_MODE = 3                # VAD敏感度 (0-3, 越高越敏感)
SILENCE_THRESHOLD = 1.0     # 无声间隔阈值(秒)

DEFAULT_CONFIG = {
    "wake_words": ["你好小智"],                  # 支持多唤醒词
    "wake_words_pinyin": ["ni hao xiao zhi"],   # 唤醒词拼音
    "enable_wake_word": False,                   # 是否启用唤醒词
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
    "exit_keywords": ["再见", "拜拜", "结束", "关闭"],  # 退出聆听状态的关键词
    "enable_streaming_recognition": True,       # 是否启用周期性流式识别
    "streaming_interval_seconds": 1.0,          # 流式识别尝试的最小时间间隔(秒)
    "min_streaming_duration_seconds": 1.5       # 进行流式识别所需的最小音频时长(秒)
}

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
        
        self.debug = debug

        if model_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.model_dir = os.path.join(base_dir, "models", "speech")
        else:
            self.model_dir = model_dir
            
        self.output_dir = os.path.join(self.model_dir, "output")
        self.speaker_db_dir = os.path.join(self.model_dir, "speaker_db")
        self.temp_speaker_dir = os.path.join(self.model_dir, "temp_speakers")
        self.model_cache_dir = os.path.join(self.model_dir, "model_cache")
        
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.speaker_db_dir, exist_ok=True)
        os.makedirs(self.temp_speaker_dir, exist_ok=True)
        os.makedirs(self.model_cache_dir, exist_ok=True)

        if config_path is None:
            self.config_path = os.path.join(self.model_dir, "config.json")
        else:
            self.config_path = config_path
            
        self.config = self._load_config()
        
        self.is_recording = False 
        self.is_listening = not self.config["enable_wake_word"]     # 若不启用唤醒词，则一直监听
        self.is_enrolling = False                                   # 是否处于声纹注册模式
        self.current_speaker_id = None                              # 当前识别到的说话人ID
        
        # VAD模块 (WebRTC)
        self.vad = None
        
        # 音频处理队列和缓冲区
        self.audio_queue = Queue()
        self.audio_segments = []
        self.last_voice_time = time.time()
        self.last_streaming_recognition_time = time.time()  # 用于追踪上次流式识别的时间
    
        # 模型
        self.asr_model = None
        self.sv_model = None
        
        # 注册的说话人列表
        self.speakers = {}
        self.temp_speakers = {}
        
        # 线程管理
        self.record_thread = None
        self.process_thread = None
        self._stop_event = threading.Event()
        self._state_lock = threading.Lock()
        self._latest_state = SpeechState()
        
        # 注册信息
        self.enrolling_id = None
        self.enrolling_name = None
        
    def _load_config(self) -> dict:
        """加载配置文件，若不存在则创建默认配置"""
        if not os.path.exists(self.config_path):
            logger.info(f"配置文件不存在，创建默认配置: {self.config_path}")
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(DEFAULT_CONFIG, f, ensure_ascii=False, indent=4)
            return DEFAULT_CONFIG.copy()
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                for key, value in DEFAULT_CONFIG.items():
                    if key not in config:
                        config[key] = value

                if len(config['wake_words']) != len(config['wake_words_pinyin']):
                    config['wake_words_pinyin'] = [' '.join(lazy_pinyin(word)) for word in config['wake_words']]

            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=4)
                
            return config
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}，使用默认配置")
            return DEFAULT_CONFIG.copy()
            
    def _download_and_cache_model(self, model_id: str, model_version: str = None) -> str:
        """
        下载并缓存模型，如果本地已有缓存则直接使用本地模型
        
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
                logger.info(f"正在下载模型: {model_id} {model_version if model_version else ''}")
                model_path = snapshot_download(model_id, 
                                             revision=model_version,
                                             cache_dir=model_cache_path)
                
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
                logger.error(f"模型下载失败: {e}")
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
            
    def initialize(self) -> int:
        """
        初始化语音识别模块
        
        Returns:
            int: 错误码，0表示成功，其他值表示失败
        """
        if self._is_running:
            return ALREADY_INITIALIZED
            
        try:
            self.vad = webrtcvad.Vad()
            self.vad.set_mode(VAD_MODE)
            logger.info("VAD初始化成功")

            logger.info("正在初始化语音识别模型...")

            asr_model_path = os.path.join(self.model_cache_dir, ASR_MODEL_ID.replace("/", "_"), "model_info.json")
            if os.path.exists(asr_model_path):
                logger.info("从本地缓存加载ASR模型")
                try:
                    self.asr_model = AutoModel(model=ASR_MODEL_ID, 
                                            trust_remote_code=True,
                                            model_dir=self.model_cache_dir)
                    logger.info("本地ASR模型加载成功")
                except Exception as e:
                    logger.error(f"本地ASR模型加载失败: {e}，尝试在线下载")
                    model_path = self._download_and_cache_model(ASR_MODEL_ID)
                    self.asr_model = AutoModel(model=model_path, 
                                            trust_remote_code=True)
            else:
                logger.info("在线下载ASR模型")
                model_path = self._download_and_cache_model(ASR_MODEL_ID)
                self.asr_model = AutoModel(model=ASR_MODEL_ID,
                                         trust_remote_code=True)
            
            logger.info("语音识别模型加载成功")

            if self.config["enable_speaker_verification"]:
                logger.info("正在初始化声纹识别模型...")

                sv_cache_key = f"{SV_MODEL_ID.replace('/', '_')}_{SV_MODEL_VERSION}"
                sv_model_path = os.path.join(self.model_cache_dir, sv_cache_key, "model_info.json")
                
                if os.path.exists(sv_model_path):
                    logger.info("从本地缓存加载声纹识别模型")
                    try:
                        with open(sv_model_path, 'r', encoding='utf-8') as f:
                            sv_model_info = json.load(f)
                        
                        self.sv_model = pipeline(
                            task='speaker-verification',
                            model=SV_MODEL_ID,
                            model_revision=SV_MODEL_VERSION,
                            cache_dir=self.model_cache_dir
                        )
                        logger.info("本地声纹识别模型加载成功")
                    except Exception as e:
                        logger.error(f"本地声纹识别模型加载失败: {e}，尝试在线下载")
                        model_path = self._download_and_cache_model(SV_MODEL_ID, SV_MODEL_VERSION)
                        self.sv_model = pipeline(
                            task='speaker-verification',
                            model=SV_MODEL_ID,
                            model_revision=SV_MODEL_VERSION,
                            cache_dir=self.model_cache_dir
                        )
                else:
                    logger.info("在线下载声纹识别模型")
                    model_path = self._download_and_cache_model(SV_MODEL_ID, SV_MODEL_VERSION)
                    self.sv_model = pipeline(
                        task='speaker-verification',
                        model=SV_MODEL_ID,
                        model_revision=SV_MODEL_VERSION,
                        cache_dir=self.model_cache_dir
                    )
                
                logger.info("声纹识别模型加载成功")
            
            self._load_speakers()
            self._load_temp_speakers()

            with self.audio_queue.mutex:
                self.audio_queue.queue.clear()
            self.audio_segments = []
            
            self.is_recording = False
            self.is_listening = not self.config["enable_wake_word"]
            self.is_enrolling = False
            self.current_speaker_id = None
            
            logger.info("语音识别模块初始化完成")
            logger.info(f"唤醒词功能: {'开启' if self.config['enable_wake_word'] else '关闭'}")
            if self.config['enable_wake_word']:
                logger.info(f"唤醒词列表: {self.config['wake_words']}")
            logger.info(f"声纹识别功能: {'开启' if self.config['enable_speaker_verification'] else '关闭'}")
            logger.info(f"已注册说话人: {len(self.speakers)}人")
            logger.info(f"临时声纹数量: {len(self.temp_speakers)}个")
            
            return SUCCESS
        except Exception as e:
            logger.error(f"初始化失败: {e}")
            return MODEL_LOADING_FAILED
            
    def _load_speakers(self):
        """加载已注册的说话人信息"""
        self.speakers = {}
        speaker_files = [f for f in os.listdir(self.speaker_db_dir) if f.endswith('.wav')]
        
        for speaker_file in speaker_files:
            speaker_id = os.path.splitext(speaker_file)[0]
            speaker_path = os.path.join(self.speaker_db_dir, speaker_file)
            info_path = os.path.join(self.speaker_db_dir, f"{speaker_id}.json")
            
            if os.path.exists(info_path):
                try:
                    with open(info_path, 'r', encoding='utf-8') as f:
                        speaker_info = json.load(f)
                        self.speakers[speaker_id] = speaker_info
                except:
                    self.speakers[speaker_id] = {"name": f"用户{speaker_id}", "path": speaker_path}
            else:
                self.speakers[speaker_id] = {"name": f"用户{speaker_id}", "path": speaker_path}
                with open(info_path, 'w', encoding='utf-8') as f:
                    json.dump(self.speakers[speaker_id], f, ensure_ascii=False, indent=4)
                    
        logger.info(f"已加载 {len(self.speakers)} 个注册说话人")
        
    def _load_temp_speakers(self):
        """加载临时未识别说话人信息"""
        self.temp_speakers = {}
        temp_files = [f for f in os.listdir(self.temp_speaker_dir) if f.endswith('.wav')]
        
        for temp_file in temp_files:
            temp_id = os.path.splitext(temp_file)[0]
            temp_path = os.path.join(self.temp_speaker_dir, temp_file)
            info_path = os.path.join(self.temp_speaker_dir, f"{temp_id}.json")
            
            if os.path.exists(info_path):
                try:
                    with open(info_path, 'r', encoding='utf-8') as f:
                        speaker_info = json.load(f)
                        self.temp_speakers[temp_id] = speaker_info
                except:
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    self.temp_speakers[temp_id] = {
                        "name": f"未知用户{temp_id}", 
                        "path": temp_path,
                        "created": timestamp
                    }
            else:
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                self.temp_speakers[temp_id] = {
                    "name": f"未知用户{temp_id}", 
                    "path": temp_path,
                    "created": timestamp
                }
                with open(info_path, 'w', encoding='utf-8') as f:
                    json.dump(self.temp_speakers[temp_id], f, ensure_ascii=False, indent=4)
                    
        self._cleanup_temp_speakers()
        
        logger.info(f"已加载 {len(self.temp_speakers)} 个临时声纹")
        
    def _cleanup_temp_speakers(self):
        """清理临时声纹，保持在限制数量内"""
        max_temp = self.config.get("max_temp_speakers", 10)
        if len(self.temp_speakers) <= max_temp:
            return
            
        sorted_temps = sorted(
            self.temp_speakers.items(), 
            key=lambda x: x[1].get("created", "")
        )
        
        to_delete = sorted_temps[:(len(self.temp_speakers) - max_temp)]
        for temp_id, _ in to_delete:
            self.delete_temp_speaker(temp_id)
            
    def update(self) -> Optional[SpeechState]:
        """
        获取最新的语音识别状态
        
        Returns:
            Optional[SpeechState]: 当前语音识别状态或None
        """
        if not self._is_running:
            return None
            
        with self._state_lock:
            return self._latest_state
            
    def check_vad_activity(self, audio_data: np.ndarray) -> bool:
        """
        检查音频段中是否有语音活动
        
        Args:
            audio_data: 音频数据numpy数组
            
        Returns:
            bool: 是否检测到语音活动
        """
        step_ms = 20  # 20ms
        step_size = int(RATE * step_ms / 1000)  # 每20ms的样本数
        voice_frames = 0
        total_frames = 0
        
        # 拆分音频为20ms的小段进行检测
        for i in range(0, len(audio_data) - step_size, step_size):
            frame = audio_data[i:i+step_size]
            if len(frame) == step_size:
                total_frames += 1
                if self.vad.is_speech(frame.tobytes(), sample_rate=RATE):
                    voice_frames += 1
        
        if total_frames == 0:
            return False
            
        ratio = voice_frames / total_frames
        return ratio > 0.4  # 如果超过40%的帧包含语音，视为有语音活动
        
    def _save_audio(self, audio_data: bytes, filepath: str) -> str:
        """保存音频数据到文件"""
        with wave.open(filepath, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)  # 16-bit PCM = 2 bytes
            wf.setframerate(RATE)
            wf.writeframes(audio_data)
        return filepath
        
    def process_audio_chunk(self, audio_chunk: bytes):
        """处理音频块"""
        if not self.is_recording or not self._is_running:
            return
            
        audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
        vad_active = self.check_vad_activity(audio_data)
        
        if vad_active:
            self.audio_segments.append(audio_chunk)
            self.last_voice_time = time.time()

            if self.config.get("enable_streaming_recognition", False) and \
               (self.is_listening or not self.config["enable_wake_word"]):
                
                current_audio_duration = len(b''.join(self.audio_segments)) / (RATE * CHANNELS * 2)
                if current_audio_duration >= self.config.get("min_streaming_duration_seconds", 1.5) and \
                   (time.time() - self.last_streaming_recognition_time) >= self.config.get("streaming_interval_seconds", 1.0):
                    
                    self._perform_streaming_recognition()
                    self.last_streaming_recognition_time = time.time()

            elif not self.is_listening and self.config["enable_wake_word"] and len(self.audio_segments) == 1:
                logger.debug("检测到声音，等待唤醒词...")
        else:
            if self.audio_segments and (time.time() - self.last_voice_time) > SILENCE_THRESHOLD:
                self._process_audio_segments()
                self.last_streaming_recognition_time = time.time()
            
    def _perform_streaming_recognition(self):
        """执行周期性的流式识别，更新部分状态（主要是文本和关键词）"""
        if not self.audio_segments:
            return

        current_audio_data = b''.join(self.audio_segments)
        if (len(current_audio_data) / (RATE * CHANNELS * 2)) < self.config.get("min_streaming_duration_seconds", 1.5):
            return

        stream_temp_file = os.path.join(self.output_dir, "stream_temp_audio.wav")
        try:
            self._save_audio(current_audio_data, stream_temp_file)

            logger.debug(f"执行流式识别，音频时长: {len(current_audio_data) / (RATE * CHANNELS * 2):.2f}s.")
            text = self._recognize_speech(stream_temp_file)

            if text:
                with self._state_lock:
                    self._latest_state.recognition["text"] = text
                    
                    keyword, category = self._detect_keyword(text)
                    if keyword:
                        self._latest_state.recognition["keyword"] = keyword
                        self._latest_state.recognition["keyword_category"] = category
                        self._latest_state.recognition["has_keyword"] = True
                    else:
                        self._latest_state.recognition["keyword"] = ""
                        self._latest_state.recognition["keyword_category"] = ""
                        self._latest_state.recognition["has_keyword"] = False
                    
                    self._latest_state.timestamp = time.time()
        except Exception as e:
            logger.error(f"流式识别出错: {e}")
        finally:
            if os.path.exists(stream_temp_file):
                try:
                    os.remove(stream_temp_file)
                except Exception as e:
                    logger.warning(f"无法删除流式临时文件 {stream_temp_file}: {e}")
        
    def _recognize_speech(self, audio_file: str) -> str:
        """
        使用SenceVoice识别语音内容
        
        Args:
            audio_file: 音频文件路径
            
        Returns:
            str: 识别结果文本，失败返回空字符串
        """
        try:
            import sys
            import io
            original_stdout = sys.stdout
            sys.stdout = io.StringIO()
            
            try:
                res = self.asr_model.generate(
                    input=audio_file,
                    cache={},
                    language="zn",  # "zn", "en", "yue", "ja", "ko", "nospeech", "auto"
                    use_itn=False,
                )
                
                recognized_text = res[0]['text'].split(">")[-1]
                return recognized_text
            finally:
                sys.stdout = original_stdout
        except Exception as e:
            logger.error(f"语音识别失败: {e}")
            return ""
            
    def _identify_speaker(self, audio_file: str) -> tuple:
        """
        进行说话人识别
        
        Args:
            audio_file: 音频文件路径
            
        Returns:
            tuple: (speaker_id, is_registered, similarity)
                   speaker_id: 识别到的说话人ID
                   is_registered: 是否是注册过的说话人
                   similarity: 相似度分数
        """
        if not self.config["enable_speaker_verification"]:
            return None, False, 0.0
            
        try:
            best_score = -1
            best_speaker = None
            is_registered = False
            threshold = self.config["speaker_verification_threshold"]
            
            for speaker_id, info in self.speakers.items():
                speaker_file = info["path"]

                result = self.sv_model([speaker_file, audio_file])
                similarity = result["score"]
                
                if similarity > threshold and similarity > best_score:
                    best_score = similarity
                    best_speaker = speaker_id
                    is_registered = True
            
            if best_speaker is None:
                for temp_id, info in self.temp_speakers.items():
                    temp_file = info["path"]
                    
                    result = self.sv_model([temp_file, audio_file])
                    similarity = result["score"]
                    
                    if similarity > threshold and similarity > best_score:
                        best_score = similarity
                        best_speaker = temp_id
                        is_registered = False
            
            return best_speaker, is_registered, best_score
        except Exception as e:
            logger.error(f"声纹识别失败: {e}")
            return None, False, 0.0
            
    def _detect_wake_word(self, text: str) -> str:
        """
        检测文本中是否包含唤醒词
        
        Args:
            text: 需要检测的文本
            
        Returns:
            str: 检测到的唤醒词，未检测到则返回空字符串
        """
        if not text or not self.config["enable_wake_word"]:
            return ""
            
        text_pinyin = ' '.join(lazy_pinyin(text.lower()))
        
        for i, wake_word in enumerate(self.config["wake_words"]):
            wake_pinyin = self.config["wake_words_pinyin"][i]
            if wake_pinyin in text_pinyin:
                return wake_word
                
        return ""
        
    def _detect_keyword(self, text: str) -> tuple:
        """
        检测文本中是否包含特定关键词
        
        Args:
            text: 需要检测的文本
            
        Returns:
            tuple: (keyword, category)
                   keyword: 检测到的关键词
                   category: 关键词所属类别
        """
        if not text:
            return "", ""
            
        keywords = self.config.get("keywords", {})
        for keyword, category in keywords.items():
            if keyword in text:
                return keyword, category
                
        return "", ""
        
    def _process_audio_segments(self):
        """处理收集到的音频段落"""
        if not self.audio_segments:
            return
            
        audio_data = b''.join(self.audio_segments)
        audio_duration = len(audio_data) / (RATE * CHANNELS * 2)

        if self.is_enrolling:
            if audio_duration < self.config["min_enrollment_duration"]:
                logger.info(f"声纹注册语音太短 ({audio_duration:.1f}秒)，需要至少 {self.config['min_enrollment_duration']} 秒")
                self.audio_segments = []
                return
                
            enrollment_path = os.path.join(self.speaker_db_dir, f"{self.enrolling_id}.wav")
            self._save_audio(audio_data, enrollment_path)
  
            speaker_info = {
                "name": self.enrolling_name,
                "path": enrollment_path,
                "created": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            with open(os.path.join(self.speaker_db_dir, f"{self.enrolling_id}.json"), 'w', encoding='utf-8') as f:
                json.dump(speaker_info, f, ensure_ascii=False, indent=4)
                
            self.speakers[self.enrolling_id] = speaker_info
            
            logger.info(f"声纹注册成功: {self.enrolling_name} (ID: {self.enrolling_id})")
            self.is_enrolling = False
            self.audio_segments = []
            
            state = SpeechState()
            state.recognition["text"] = f"声纹注册成功: {self.enrolling_name}"
            
            with self._state_lock:
                self._latest_state = state
                
            return

        temp_file = os.path.join(self.output_dir, "temp_audio.wav")
        self._save_audio(audio_data, temp_file)
        
        state = SpeechState()
        if not self.is_listening and self.config["enable_wake_word"]:
            text = self._recognize_speech(temp_file)
            state.recognition["text"] = text
            
            if text:
                wake_word = self._detect_wake_word(text)
                if wake_word:
                    logger.info(f"检测到唤醒词: {wake_word}")
                    self.is_listening = True
                    
                    state.recognition["wake_word"] = wake_word
                    state.recognition["has_wake_word"] = True
                    state.recognition["is_command"] = True

                    if self.config["enable_speaker_verification"]:
                        speaker_id, is_registered, confidence = self._identify_speaker(temp_file)
                        if speaker_id:
                            self.current_speaker_id = speaker_id
                            speaker_info = self.speakers.get(speaker_id) if is_registered else self.temp_speakers.get(speaker_id)
                            speaker_name = speaker_info["name"] if speaker_info else f"用户{speaker_id}"
                            
                            state.recognition["speaker_id"] = speaker_id
                            state.recognition["speaker_name"] = speaker_name
                            state.recognition["is_registered_speaker"] = is_registered
                            state.recognition["confidence"] = confidence
                            
                            logger.info(f"识别到说话人: {speaker_name} {'(已注册)' if is_registered else '(临时)'}")
                else:
                    self.audio_segments = []
                    return
        else:
            text = self._recognize_speech(temp_file)
            state.recognition["text"] = text
            
            if text:
                wake_word = self._detect_wake_word(text)
                if wake_word:
                    state.recognition["wake_word"] = wake_word
                    state.recognition["has_wake_word"] = True
                    state.recognition["is_command"] = True
                
                keyword, category = self._detect_keyword(text)
                if keyword:
                    logger.info(f"检测到关键词: {keyword}, 类别: {category}")
                    state.recognition["keyword"] = keyword
                    state.recognition["keyword_category"] = category
                    state.recognition["has_keyword"] = True
                    state.recognition["is_command"] = True
                
                if self.config["enable_speaker_verification"]:
                    speaker_id, is_registered, confidence = self._identify_speaker(temp_file)
                    
                    if speaker_id:
                        self.current_speaker_id = speaker_id
                        speaker_info = self.speakers.get(speaker_id) if is_registered else self.temp_speakers.get(speaker_id)
                        speaker_name = speaker_info["name"] if speaker_info else f"用户{speaker_id}"
                        
                        state.recognition["speaker_id"] = speaker_id
                        state.recognition["speaker_name"] = speaker_name
                        state.recognition["is_registered_speaker"] = is_registered
                        state.recognition["confidence"] = confidence
                        
                        logger.info(f"识别到说话人: {speaker_name} {'(已注册)' if is_registered else '(临时)'}")
                    else:
                        self._save_temp_speaker(temp_file)
                
                if self.config["enable_wake_word"] and self.is_listening:
                    exit_keywords = self.config.get("exit_keywords", ["再见", "拜拜", "结束", "关闭"])
                    if any(exit_word in text for exit_word in exit_keywords):
                        logger.info(f"检测到结束指令：{text}，退出聆听状态")
                        self.is_listening = False
                        
        with self._state_lock:
            self._latest_state = state
        
        self.audio_segments = []
        
    def _save_temp_speaker(self, audio_file: str) -> str:
        """
        保存未识别的临时声纹
        
        Args:
            audio_file: 音频文件路径
            
        Returns:
            str: 临时声纹ID
        """
        temp_id = f"temp_{str(uuid.uuid4())[:6]}"
        temp_path = os.path.join(self.temp_speaker_dir, f"{temp_id}.wav")
        
        shutil.copy(audio_file, temp_path)

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        info = {
            "name": f"未知用户{temp_id}",
            "path": temp_path,
            "created": timestamp
        }
        
        with open(os.path.join(self.temp_speaker_dir, f"{temp_id}.json"), 'w', encoding='utf-8') as f:
            json.dump(info, f, ensure_ascii=False, indent=4)
            
        self.temp_speakers[temp_id] = info
        self._cleanup_temp_speakers()
        
        logger.info(f"保存临时声纹: {temp_id}")
        return temp_id
        
    def _audio_recording_thread(self):
        """音频录制线程"""
        logger.info("音频录制线程启动")
        
        p = pyaudio.PyAudio()

        try:
            stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
            
            logger.info("录音开始")
            
            while not self._stop_event.is_set() and self.is_recording:
                if self.audio_queue.qsize() < 100:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    self.audio_queue.put(data)
                else:
                    time.sleep(0.01)
        except Exception as e:
            logger.error(f"录音线程错误: {e}")
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
            while not self._stop_event.is_set():
                if not self.audio_queue.empty():
                    audio_data = self.audio_queue.get()
                    self.process_audio_chunk(audio_data)
                else:
                    time.sleep(0.01)
        except Exception as e:
            logger.error(f"音频处理线程错误: {e}")
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
            logger.error(f"无法启动语音识别系统: {result}")
            return result
            
        try:
            self._stop_event.clear()
            self.is_recording = True
            
            self.record_thread = threading.Thread(target=self._audio_recording_thread)
            self.record_thread.daemon = True
            self.record_thread.start()
            
            self.process_thread = threading.Thread(target=self._audio_processing_thread)
            self.process_thread.daemon = True
            self.process_thread.start()
            
            logger.info("语音识别系统启动成功")
            if self.config["enable_wake_word"]:
                logger.info(f"请说唤醒词 {self.config['wake_words']} 来开始对话")
            else:
                logger.info("已进入语音识别模式")
                
            return SUCCESS
        except Exception as e:
            logger.error(f"启动处理线程时出错: {str(e)}")
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
            
            self._stop_event.set()
            self.is_recording = False
            
            if self.record_thread and self.record_thread.is_alive():
                self.record_thread.join(timeout=2.0)
            if self.process_thread and self.process_thread.is_alive():
                self.process_thread.join(timeout=2.0)
                
            self.record_thread = None
            self.process_thread = None

            self.asr_model = None
            self.sv_model = None
            self.vad = None
            
            with self.audio_queue.mutex:
                self.audio_queue.queue.clear()
            self.audio_segments = []
            
            logger.info("语音识别系统已关闭")
            return SUCCESS
        except Exception as e:
            logger.error(f"关闭语音识别系统失败: {str(e)}")
            return OPERATION_FAILED
            
    # ===== 公共API方法 =====
    
    def add_wake_word(self, wake_word: str) -> bool:
        """
        添加新的唤醒词
        
        Args:
            wake_word: 要添加的唤醒词
            
        Returns:
            bool: 是否添加成功
        """
        if wake_word in self.config["wake_words"]:
            logger.warning(f"唤醒词 '{wake_word}' 已存在")
            return False
            
        wake_pinyin = ' '.join(lazy_pinyin(wake_word))
        self.config["wake_words"].append(wake_word)
        self.config["wake_words_pinyin"].append(wake_pinyin)
        
        # 保存到配置文件
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, ensure_ascii=False, indent=4)
            
        logger.info(f"添加唤醒词成功: {wake_word} ({wake_pinyin})")
        return True
        
    def remove_wake_word(self, wake_word: str) -> bool:
        """
        移除唤醒词
        
        Args:
            wake_word: 要移除的唤醒词
            
        Returns:
            bool: 是否移除成功
        """
        if wake_word not in self.config["wake_words"]:
            logger.warning(f"唤醒词 '{wake_word}' 不存在")
            return False
            
        # 至少保留一个唤醒词
        if len(self.config["wake_words"]) <= 1:
            logger.warning(f"不能移除最后一个唤醒词")
            return False
            
        idx = self.config["wake_words"].index(wake_word)
        self.config["wake_words"].pop(idx)
        self.config["wake_words_pinyin"].pop(idx)
        
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, ensure_ascii=False, indent=4)
            
        logger.info(f"移除唤醒词成功: {wake_word}")
        return True
        
    def toggle_wake_word(self, enable: bool = None) -> bool:
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
            
        self.is_listening = not self.config["enable_wake_word"]
        
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, ensure_ascii=False, indent=4)
            
        logger.info(f"唤醒词功能已{'开启' if self.config['enable_wake_word'] else '关闭'}")
        return self.config["enable_wake_word"]
        
    def register_speaker(self, name: str = None) -> bool:
        """
        进入声纹注册模式
        
        Args:
            name: 要注册的说话人名称
            
        Returns:
            bool: 是否成功进入注册模式
        """
        if not self.config["enable_speaker_verification"]:
            logger.warning("声纹识别功能未开启")
            return False
            
        logger.info("进入声纹注册模式，请说话...")
        self.is_enrolling = True
        self.audio_segments = []
        
        # 生成新的说话人ID
        self.enrolling_id = str(uuid.uuid4())[:8]
        self.enrolling_name = name if name else f"用户{self.enrolling_id}"
        
        return True
        
    def delete_speaker(self, speaker_id: str) -> bool:
        """
        删除已注册的声纹
        
        Args:
            speaker_id: 说话人ID
            
        Returns:
            bool: 是否删除成功
        """
        if speaker_id not in self.speakers:
            logger.warning(f"说话人ID '{speaker_id}' 不存在")
            return False
            
        try:
            wav_path = os.path.join(self.speaker_db_dir, f"{speaker_id}.wav")
            json_path = os.path.join(self.speaker_db_dir, f"{speaker_id}.json")
            
            if os.path.exists(wav_path):
                os.remove(wav_path)
            if os.path.exists(json_path):
                os.remove(json_path)
                
            del self.speakers[speaker_id]
            
            logger.info(f"删除声纹成功: {speaker_id}")
            return True
        except Exception as e:
            logger.error(f"删除声纹失败: {e}")
            return False
            
    def delete_temp_speaker(self, temp_id: str) -> bool:
        """
        删除临时声纹
        
        Args:
            temp_id: 临时声纹ID
            
        Returns:
            bool: 是否删除成功
        """
        if temp_id not in self.temp_speakers:
            logger.warning(f"临时声纹ID '{temp_id}' 不存在")
            return False
            
        try:
            wav_path = os.path.join(self.temp_speaker_dir, f"{temp_id}.wav")
            json_path = os.path.join(self.temp_speaker_dir, f"{temp_id}.json")
            
            if os.path.exists(wav_path):
                os.remove(wav_path)
            if os.path.exists(json_path):
                os.remove(json_path)
                
            del self.temp_speakers[temp_id]
            
            logger.info(f"删除临时声纹成功: {temp_id}")
            return True
        except Exception as e:
            logger.error(f"删除临时声纹失败: {e}")
            return False
            
    def promote_temp_speaker(self, temp_id: str, name: str = None) -> str:
        """
        将临时声纹提升为正式声纹
        
        Args:
            temp_id: 临时声纹ID
            name: 新的用户名，若为None则使用临时声纹的名称
            
        Returns:
            str: 新的说话人ID或空字符串(表示失败)
        """
        if temp_id not in self.temp_speakers:
            logger.warning(f"临时声纹ID '{temp_id}' 不存在")
            return ""
            
        try:
            new_id = str(uuid.uuid4())[:8]
            
            temp_info = self.temp_speakers[temp_id]
            temp_path = temp_info["path"]
            
            if name is None:
                name = temp_info.get("name", f"用户{new_id}")
                
            new_path = os.path.join(self.speaker_db_dir, f"{new_id}.wav")
            shutil.copy(temp_path, new_path)
            
            speaker_info = {
                "name": name,
                "path": new_path,
                "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                "promoted_from": temp_id
            }
            
            with open(os.path.join(self.speaker_db_dir, f"{new_id}.json"), 'w', encoding='utf-8') as f:
                json.dump(speaker_info, f, ensure_ascii=False, indent=4)
                
            self.speakers[new_id] = speaker_info
            
            self.delete_temp_speaker(temp_id)
            
            logger.info(f"提升临时声纹成功: {temp_id} -> {new_id} ({name})")
            return new_id
        except Exception as e:
            logger.error(f"提升临时声纹失败: {e}")
            return ""
            
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
        
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, ensure_ascii=False, indent=4)
            
        logger.info(f"添加关键词成功: '{keyword}' -> {category}")
        return True
        
    def remove_keyword(self, keyword: str) -> bool:
        """
        删除特定关键词
        
        Args:
            keyword: 要删除的关键词
            
        Returns:
            bool: 是否删除成功
        """
        if "keywords" not in self.config or keyword not in self.config["keywords"]:
            logger.warning(f"关键词 '{keyword}' 不存在")
            return False
            
        del self.config["keywords"][keyword]
        
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, ensure_ascii=False, indent=4)
            
        logger.info(f"删除关键词成功: '{keyword}'")
        return True
        
    def get_registered_speakers(self) -> Dict[str, Dict]:
        """
        获取所有注册的说话人信息
        
        Returns:
            Dict[str, Dict]: 说话人ID到信息的映射
        """
        return self.speakers.copy()
        
    def get_temp_speakers(self) -> Dict[str, Dict]:
        """
        获取所有临时说话人信息
        
        Returns:
            Dict[str, Dict]: 临时说话人ID到信息的映射
        """
        return self.temp_speakers.copy()
        
    def set_max_temp_speakers(self, count: int) -> bool:
        """
        设置最大临时声纹数量
        
        Args:
            count: 最大临时声纹数量
            
        Returns:
            bool: 是否设置成功
        """
        if count < 1:
            logger.warning(f"最大临时声纹数量不能小于1")
            return False
            
        self.config["max_temp_speakers"] = count
        
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, ensure_ascii=False, indent=4)
            
        self._cleanup_temp_speakers()
        
        logger.info(f"设置最大临时声纹数量: {count}")
        return True
    
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
            logger.warning(f"退出关键词 '{keyword}' 已存在")
            return False
            
        self.config["exit_keywords"].append(keyword)
        
        # 保存到配置文件
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, ensure_ascii=False, indent=4)
            
        logger.info(f"添加退出关键词成功: {keyword}")
        return True
        
    def remove_exit_keyword(self, keyword: str) -> bool:
        """
        删除退出聆听状态的关键词
        
        Args:
            keyword: 要删除的关键词
            
        Returns:
            bool: 是否删除成功
        """
        if "exit_keywords" not in self.config or keyword not in self.config["exit_keywords"]:
            logger.warning(f"退出关键词 '{keyword}' 不存在")
            return False
            
        # 至少保留一个退出关键词
        if len(self.config["exit_keywords"]) <= 1:
            logger.warning(f"不能删除最后一个退出关键词")
            return False
            
        self.config["exit_keywords"].remove(keyword)
        
        # 保存到配置文件
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, ensure_ascii=False, indent=4)
            
        logger.info(f"删除退出关键词成功: {keyword}")
        return True
        
    def get_exit_keywords(self) -> List[str]:
        """
        获取所有退出聆听状态的关键词
        
        Returns:
            List[str]: 退出关键词列表
        """
        return self.config.get("exit_keywords", ["再见", "拜拜", "结束", "关闭"]).copy()

    def enable_listening(self) -> bool:
        """
        启用聆听状态
        
        Returns:
            bool: 是否启用成功
        """
        if self.is_listening:
            logger.warning("已处于聆听状态")
            return False
            
        self.is_listening = True
        logger.info("已启用聆听状态")
        return True
    
    def disable_listening(self) -> bool:
        """
        禁用聆听状态
        
        Returns:
            bool: 是否禁用成功
        """
        if not self.is_listening:
            logger.warning("已处于非聆听状态")
            return False
            
        self.is_listening = False
        logger.info("已禁用聆听状态")
        return True
    
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
    