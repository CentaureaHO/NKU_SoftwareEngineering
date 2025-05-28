"""语音识别说话人管理模块，负责管理注册说话人和临时说话人"""

import json
import logging
import os
import shutil
import time
import uuid
from typing import Dict, Optional, Tuple

logger = logging.getLogger('SpeakerManager')

class SpeakerManager:
    """说话人管理类，负责管理注册说话人和临时说话人"""

    def __init__(self, speaker_db_dir: str, temp_speaker_dir: str, max_temp_speakers: int = 10):
        """
        初始化说话人管理器
        
        Args:
            speaker_db_dir: 注册说话人存储目录
            temp_speaker_dir: 临时说话人存储目录
            max_temp_speakers: 最大临时说话人数量
        """
        self.speaker_db_dir = speaker_db_dir
        self.temp_speaker_dir = temp_speaker_dir
        self.max_temp_speakers = max_temp_speakers

        os.makedirs(self.speaker_db_dir, exist_ok=True)
        os.makedirs(self.temp_speaker_dir, exist_ok=True)

        self.speakers = {}  # 注册说话人
        self.temp_speakers = {}  # 临时说话人

        self._load_speakers()
        self._load_temp_speakers()

    def _load_speakers(self):
        """加载已注册的说话人信息"""
        self.speakers = {}
        speaker_files = [f for f in os.listdir(
            self.speaker_db_dir) if f.endswith('.wav')]

        for speaker_file in speaker_files:
            speaker_id = os.path.splitext(speaker_file)[0]
            speaker_path = os.path.join(self.speaker_db_dir, speaker_file)
            info_path = os.path.join(self.speaker_db_dir, f"{speaker_id}.json")

            if os.path.exists(info_path):
                try:
                    with open(info_path, 'r', encoding='utf-8') as f:
                        speaker_info = json.load(f)
                        self.speakers[speaker_id] = speaker_info
                except (json.JSONDecodeError, FileNotFoundError, PermissionError) as e:
                    logger.warning("读取说话人信息失败: %s, 使用默认信息", e)
                    self.speakers[speaker_id] = {
                        "name": f"用户{speaker_id}", "path": speaker_path}
            else:
                self.speakers[speaker_id] = {
                    "name": f"用户{speaker_id}", "path": speaker_path}
                with open(info_path, 'w', encoding='utf-8') as f:
                    json.dump(self.speakers[speaker_id],
                              f, ensure_ascii=False, indent=4)

        logger.info("已加载 %d 个注册说话人", len(self.speakers))

    def _load_temp_speakers(self):
        """加载临时未识别说话人信息"""
        self.temp_speakers = {}
        temp_files = [f for f in os.listdir(
            self.temp_speaker_dir) if f.endswith('.wav')]

        for temp_file in temp_files:
            temp_id = os.path.splitext(temp_file)[0]
            temp_path = os.path.join(self.temp_speaker_dir, temp_file)
            info_path = os.path.join(self.temp_speaker_dir, f"{temp_id}.json")

            if os.path.exists(info_path):
                try:
                    with open(info_path, 'r', encoding='utf-8') as f:
                        speaker_info = json.load(f)
                        self.temp_speakers[temp_id] = speaker_info
                except (json.JSONDecodeError, FileNotFoundError, PermissionError) as e:
                    logger.warning("读取临时说话人信息失败: %s, 使用默认信息", e)
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
                    json.dump(
                        self.temp_speakers[temp_id], f, ensure_ascii=False, indent=4)

        self._cleanup_temp_speakers()
        logger.info("已加载 %d 个临时声纹", len(self.temp_speakers))

    def _cleanup_temp_speakers(self):
        """清理临时声纹，保持在限制数量内"""
        if len(self.temp_speakers) <= self.max_temp_speakers:
            return

        sorted_temps = sorted(
            self.temp_speakers.items(),
            key=lambda x: x[1].get("created", "")
        )

        to_delete = sorted_temps[:(len(self.temp_speakers) - self.max_temp_speakers)]
        for temp_id, _ in to_delete:
            self.delete_temp_speaker(temp_id)

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

        self.max_temp_speakers = count
        self._cleanup_temp_speakers()
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
            logger.warning("说话人ID '%s' 不存在", speaker_id)
            return False

        try:
            wav_path = os.path.join(self.speaker_db_dir, f"{speaker_id}.wav")
            json_path = os.path.join(self.speaker_db_dir, f"{speaker_id}.json")

            if os.path.exists(wav_path):
                os.remove(wav_path)
            if os.path.exists(json_path):
                os.remove(json_path)

            del self.speakers[speaker_id]
            logger.info("删除声纹成功: %s", speaker_id)
            return True
        except (FileNotFoundError, PermissionError, OSError) as e:
            logger.error("删除声纹失败: %s", e)
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
            logger.warning("临时声纹ID '%s' 不存在", temp_id)
            return False

        try:
            wav_path = os.path.join(self.temp_speaker_dir, f"{temp_id}.wav")
            json_path = os.path.join(self.temp_speaker_dir, f"{temp_id}.json")

            if os.path.exists(wav_path):
                os.remove(wav_path)
            if os.path.exists(json_path):
                os.remove(json_path)

            del self.temp_speakers[temp_id]
            logger.info("删除临时声纹成功: %s", temp_id)
            return True
        except (FileNotFoundError, PermissionError, OSError) as e:
            logger.error("删除临时声纹失败: %s", e)
            return False

    def promote_temp_speaker(self, temp_id: str, name: Optional[str] = None) -> str:
        """
        将临时声纹提升为正式声纹
        
        Args:
            temp_id: 临时声纹ID
            name: 新的用户名，若为None则使用临时声纹的名称
            
        Returns:
            str: 新的说话人ID或空字符串(表示失败)
        """
        if temp_id not in self.temp_speakers:
            logger.warning("临时声纹ID '%s' 不存在", temp_id)
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

            with open(os.path.join(self.speaker_db_dir, f"{new_id}.json"),
                      'w',
                      encoding='utf-8') as f:
                json.dump(speaker_info, f, ensure_ascii=False, indent=4)

            self.speakers[new_id] = speaker_info
            self.delete_temp_speaker(temp_id)

            logger.info("提升临时声纹成功: %s -> %s (%s)", temp_id, new_id, name)
            return new_id
        except (FileNotFoundError, PermissionError, OSError, shutil.Error) as e:
            logger.error("提升临时声纹失败: %s", e)
            return ""

    def save_temp_speaker(self, audio_file: str) -> str:
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

        with open(os.path.join(self.temp_speaker_dir, f"{temp_id}.json"),
                  'w',
                  encoding='utf-8') as f:
            json.dump(info, f, ensure_ascii=False, indent=4)

        self.temp_speakers[temp_id] = info
        self._cleanup_temp_speakers()

        logger.info("保存临时声纹: %s (%s)", temp_id, temp_path)
        return temp_id

    def register_speaker(self, audio_file: str, name: Optional[str] = None) -> Tuple[str, str]:
        """
        注册新说话人
        
        Args:
            audio_file: 音频文件路径
            name: 说话人名称，若为None则自动生成
            
        Returns:
            Tuple[str, str]: (说话人ID, 说话人名称)
        """
        speaker_id = str(uuid.uuid4())[:8]
        speaker_name = name if name else f"用户{speaker_id}"

        speaker_path = os.path.join(self.speaker_db_dir, f"{speaker_id}.wav")
        shutil.copy(audio_file, speaker_path)

        speaker_info = {
            "name": speaker_name,
            "path": speaker_path,
            "created": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        with open(os.path.join(self.speaker_db_dir, f"{speaker_id}.json"),
                  'w',
                  encoding='utf-8') as f:
            json.dump(speaker_info, f, ensure_ascii=False, indent=4)

        self.speakers[speaker_id] = speaker_info
        logger.info("注册新说话人成功: %s (%s)", speaker_name, speaker_id)

        return speaker_id, speaker_name

    def get_speakers(self) -> Dict:
        """获取所有注册的说话人"""
        return self.speakers.copy()

    def get_temp_speakers(self) -> Dict:
        """获取所有临时说话人"""
        return self.temp_speakers.copy()

    def generate_speaker_id(self) -> str:
        """
        生成一个新的说话人ID
        
        Returns:
            str: 生成的说话人ID
        """
        return str(uuid.uuid4())[:8]
