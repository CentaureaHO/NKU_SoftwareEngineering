"""语音处理模块，负责音频分析、语音识别和说话人识别"""

import io
import logging
import os
import sys
import wave
from typing import Dict, Optional, Tuple, NamedTuple, List

import numpy as np
from pypinyin import lazy_pinyin

from modality.speech.speech_state import SpeechState

logger = logging.getLogger('SpeechProcessor')


# 定义数据类型用于减少参数和局部变量
class AudioConfig(NamedTuple):
    """音频配置类"""
    rate: int
    channels: int


class SpeakerMatch(NamedTuple):
    """说话人匹配结果"""
    speaker_id: Optional[str]
    is_registered: bool
    similarity: float


class SpeechProcessor:
    """语音处理类，负责音频分析、语音识别和说话人识别"""

    def __init__(self, config: Dict, dirs: Dict, models: Dict):
        """
        初始化语音处理器

        Args:
            config: 配置字典
            dirs: 目录路径字典
            models: 模型对象字典
        """
        self.config = config
        self._dirs = dirs
        self._models = models

    def check_vad_activity(self, audio_data: np.ndarray, rate: int) -> bool:
        """
        检查音频段中是否有语音活动

        Args:
            audio_data: 音频数据numpy数组
            rate: 采样率

        Returns:
            bool: 是否检测到语音活动
        """
        if self._models["vad"] is None:
            return False

        step_ms = 20  # 20ms
        step_size = int(rate * step_ms / 1000)  # 每20ms的样本数
        voice_frames = 0
        total_frames = 0

        # 拆分音频为20ms的小段进行检测
        for i in range(0, len(audio_data) - step_size, step_size):
            frame = audio_data[i:i+step_size]
            if len(frame) == step_size:
                total_frames += 1
                if self._models["vad"].is_speech(frame.tobytes(), sample_rate=rate):
                    voice_frames += 1

        if total_frames == 0:
            return False

        ratio = voice_frames / total_frames
        return ratio > 0.4  # 如果超过40%的帧包含语音，视为有语音活动

    def save_audio(self, audio_data: bytes, filepath: str, rate: int, channels: int) -> str:
        """
        保存音频数据到文件

        Args:
            audio_data: 音频数据
            filepath: 保存路径
            rate: 采样率
            channels: 声道数

        Returns:
            str: 保存的文件路径
        """
        with wave.open(filepath, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)  # 16-bit PCM = 2 bytes
            wf.setframerate(rate)
            wf.writeframes(audio_data)
        return filepath

    def recognize_speech(self, audio_file: str) -> str:
        """
        识别音频文件中的语音内容

        Args:
            audio_file: 音频文件路径

        Returns:
            str: 识别的文本，失败则返回空字符串
        """
        if self._models["asr_model"] is None:
            return ""

        try:
            # 重定向stdout以避免模型输出过多日志
            original_stdout = sys.stdout
            sys.stdout = io.StringIO()

            try:
                res = self._models["asr_model"].generate(
                    input=audio_file,
                    cache={},
                    language="zn",  # "zn", "en", "yue", "ja", "ko", "nospeech", "auto"
                    use_itn=True,
                    batch_size_s=60,
                    merge_vad=True,  #
                    merge_length_s=15,
                )

                recognized_text = res[0]['text'].split(">")[-1]
                return recognized_text
            finally:
                sys.stdout = original_stdout
        except (RuntimeError, FileNotFoundError, IndexError, KeyError, OSError) as e:
            logger.error("语音识别失败: %s", e)
            return ""

    def _compare_speaker_files(self, reference_files: List[Tuple[str, str, bool]],
                               audio_file: str, threshold: float) -> SpeakerMatch:
        """
        比较说话人音频文件相似度

        Args:
            reference_files: 参考文件列表，每项为(说话人ID, 文件路径, 是否已注册)
            audio_file: 待比较的音频文件
            threshold: 相似度阈值

        Returns:
            SpeakerMatch: 匹配结果
        """
        best_score = -1
        best_speaker = None
        is_registered = False

        for speaker_id, file_path, registered in reference_files:
            try:
                # 修改为正确的调用方式
                result = self._models["sv_model"]([file_path, audio_file])
                similarity = result["score"]

                if similarity > threshold and similarity > best_score:
                    best_score = similarity
                    best_speaker = speaker_id
                    is_registered = registered
            except (KeyError, AttributeError) as e:
                logger.warning("比较说话人文件失败: %s - %s", speaker_id, e)

        return SpeakerMatch(best_speaker, is_registered, best_score)

    def identify_speaker(self, audio_file: str) -> SpeakerMatch:
        """
        识别音频中的说话人

        Args:
            audio_file: 音频文件路径

        Returns:
            SpeakerMatch: (说话人ID, 是否已注册, 相似度分数)
        """
        if not self.config["enable_speaker_verification"] or self._models["sv_model"] is None:
            return SpeakerMatch(None, False, 0.0)

        try:
            threshold = self.config["speaker_verification_threshold"]
            speakers = self._models.get("speakers", {})
            temp_speakers = self._models.get("temp_speakers", {})

            # 准备比较文件列表
            reference_files = []

            # 添加注册说话人
            for speaker_id, info in speakers.items():
                reference_files.append((speaker_id, info["path"], True))

            # 添加临时说话人
            for temp_id, info in temp_speakers.items():
                reference_files.append((temp_id, info["path"], False))

            # 比较所有文件
            return self._compare_speaker_files(reference_files, audio_file, threshold)

        except (RuntimeError, FileNotFoundError, OSError) as e:
            logger.error("声纹识别失败: %s", e)
            return SpeakerMatch(None, False, 0.0)

    def detect_wake_word(self, text: str) -> str:
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

    def detect_keyword(self, text: str) -> Tuple[str, str]:
        """
        检测文本中是否包含关键词

        Args:
            text: 需要检测的文本

        Returns:
            Tuple[str, str]: (关键词, 类别)
        """
        if not text:
            return "", ""

        keywords = self.config.get("keywords", {})
        for keyword, category in keywords.items():
            if keyword in text:
                return keyword, category

        return "", ""

    def process_enrollment(
        self,
        audio_data: bytes,
        enrolling_config: Dict
    ) -> Optional[SpeechState]:
        """
        处理声纹注册

        Args:
            audio_data: 音频数据
            enrolling_config: 注册配置，包含：
                - name: 注册的用户名
                - audio_config: 音频配置对象
                - speaker_manager: 说话人管理器对象

        Returns:
            Optional[SpeechState]: 处理后的状态，None表示处理失败
        """
        name = enrolling_config["name"]
        audio_config = enrolling_config["audio_config"]
        speaker_manager = enrolling_config["speaker_manager"]

        audio_duration = len(audio_data) / \
            (audio_config.rate * audio_config.channels * 2)

        if audio_duration < self.config["min_enrollment_duration"]:
            logger.warning("声纹注册语音太短 (%.1f秒)，需要至少 %.1f 秒",
                           audio_duration, self.config["min_enrollment_duration"])
            return None

        # 保存临时文件用于注册
        temp_file = os.path.join(self._dirs["output_dir"], "temp_audio.wav")
        self.save_audio(audio_data, temp_file,
                        audio_config.rate, audio_config.channels)

        # 使用说话人管理器注册说话人
        speaker_id, speaker_name = speaker_manager.register_speaker(
            temp_file,
            name
        )

        logger.info("声纹注册成功: %s (%s)", speaker_name, speaker_id)

        state = SpeechState()
        state.recognition["text"] = f"声纹注册成功: {speaker_name}"
        state.recognition["speaker_id"] = speaker_id
        state.recognition["speaker_name"] = speaker_name
        return state
