"""语音识别模型管理模块，负责模型的下载、缓存和加载"""

import json
import logging
import os
import shutil
import time
from typing import Dict, Optional, Tuple

from modelscope.hub.snapshot_download import snapshot_download

logger = logging.getLogger('ModelManager')

class ModelManager:
    """语音识别模型管理类，负责模型的下载、缓存和加载"""

    def __init__(self, model_cache_dir: str):
        """
        初始化模型管理器
        
        Args:
            model_cache_dir: 模型缓存目录
        """
        self.model_cache_dir = model_cache_dir
        os.makedirs(self.model_cache_dir, exist_ok=True)

    def get_model_paths(
        self,
        model_id: str,
        model_version: Optional[str] = None
    ) -> Dict[str, str]:
        """
        获取模型相关路径
        
        Args:
            model_id: 模型ID
            model_version: 模型版本（可选）
            
        Returns:
            Dict[str, str]: 包含模型路径信息的字典
        """
        cache_key = f"{model_id.replace('/', '_')}"
        if model_version:
            cache_key += f"_{model_version}"

        model_cache_path = os.path.join(self.model_cache_dir, cache_key)
        model_info_path = os.path.join(model_cache_path, "model_info.json")
        model_lock_path = os.path.join(model_cache_path, ".lock")

        return {
            "cache_path": model_cache_path,
            "info_path": model_info_path,
            "lock_path": model_lock_path
        }

    def _read_model_info(self, model_info_path: str) -> Optional[Dict]:
        """
        读取模型信息
        
        Args:
            model_info_path: 模型信息文件路径
            
        Returns:
            Optional[Dict]: 模型信息字典，读取失败则返回None
        """
        try:
            with open(model_info_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, PermissionError) as e:
            logger.warning("读取本地模型信息失败: %s", e)
            return None

    def _clean_corrupted_cache(self, model_cache_path: str, model_lock_path: str) -> bool:
        """
        清理损坏的模型缓存
        
        Args:
            model_cache_path: 模型缓存路径
            model_lock_path: 模型锁文件路径
            
        Returns:
            bool: 清理是否成功
        """
        if not os.path.exists(model_lock_path):
            try:
                logger.info("尝试清理损坏的模型缓存")
                shutil.rmtree(model_cache_path, ignore_errors=True)
                os.makedirs(model_cache_path, exist_ok=True)
                return True
            except (PermissionError, OSError) as cleanup_err:
                logger.error("清理损坏的模型缓存失败: %s", cleanup_err)
                return False
        return False

    def _check_model_lock(self, model_lock_path: str) -> Tuple[bool, str]:
        """
        检查模型锁状态
        
        Args:
            model_lock_path: 模型锁文件路径
            
        Returns:
            Tuple[bool, str]: (是否可以继续, 错误信息)
        """
        if not os.path.exists(model_lock_path):
            return True, ""

        lock_modified_time = os.path.getmtime(model_lock_path)
        if time.time() - lock_modified_time > 1800:  # 30分钟
            logger.warning("发现过期的锁文件，可能是之前的下载异常终止")
            try:
                os.remove(model_lock_path)
                return True, ""
            except (PermissionError, OSError) as lock_err:
                error_msg = f"模型缓存目录被锁定，且无法释放锁。请手动删除: {model_lock_path}"
                logger.error("%s: %s", error_msg, lock_err)
                return False, error_msg
        else:
            logger.warning("另一个进程正在下载模型，等待下载完成...")
            for _ in range(60):
                time.sleep(10)
                if not os.path.exists(model_lock_path):
                    return True, ""

            error_msg = f"等待其他进程下载模型超时。如果确定没有其他下载正在进行，请手动删除锁文件: {model_lock_path}"
            return False, error_msg

    def _check_local_cache(self, model_id: str, paths: Dict[str, str]) -> Optional[str]:
        """
        检查本地缓存是否有可用的模型
        
        Args:
            model_id: 模型ID
            paths: 模型路径字典
            
        Returns:
            Optional[str]: 模型路径，如果没有可用缓存则返回None
        """
        model_cache_path = paths["cache_path"]
        model_info_path = paths["info_path"]
        model_lock_path = paths["lock_path"]

        # 检查本地缓存
        if os.path.exists(model_cache_path) and os.path.exists(model_info_path):
            model_info = self._read_model_info(model_info_path)
            if model_info:
                logger.info("模型已存在本地: %s (%s)", model_id, model_info.get('timestamp'))
                return model_info.get('path', model_cache_path)
            self._clean_corrupted_cache(model_cache_path, model_lock_path)

        return None

    def _download_and_save_model(
        self,
        model_id: str,
        model_version: Optional[str],
        paths: Dict[str, str]
    ) -> str:
        """
        下载并保存模型
        
        Args:
            model_id: 模型ID
            model_version: 模型版本
            paths: 模型路径字典
            
        Returns:
            str: 模型路径
            
        Raises:
            RuntimeError: 当模型下载失败时抛出
        """
        model_cache_path = paths["cache_path"]
        model_info_path = paths["info_path"]

        logger.info("正在下载模型: %s %s", model_id, model_version or "latest")
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

        logger.info("模型下载成功: %s", model_id)
        return model_path

    def download_and_cache_model(self, model_id: str, model_version: Optional[str] = None) -> str:
        """
        下载并缓存模型，如果本地已有缓存则直接使用本地模型

        Args:
            model_id: 模型ID
            model_version: 模型版本（可选）

        Returns:
            str: 模型本地缓存路径
        
        Raises:
            RuntimeError: 当模型下载失败或被锁定时抛出
        """
        paths = self.get_model_paths(model_id, model_version)

        # 检查本地缓存
        cached_path = self._check_local_cache(model_id, paths)
        if cached_path:
            return cached_path

        # 检查锁文件
        can_proceed, error_msg = self._check_model_lock(paths["lock_path"])
        if not can_proceed:
            raise RuntimeError(error_msg)

        # 检查是否下载完成（可能由另一个进程完成）
        if os.path.exists(paths["info_path"]):
            model_info = self._read_model_info(paths["info_path"])
            if model_info:
                logger.info("模型已由另一进程下载完成: %s", model_id)
                return model_info.get('path', paths["cache_path"])

        # 创建模型缓存目录
        os.makedirs(paths["cache_path"], exist_ok=True)

        # 创建锁文件并下载模型
        try:
            with open(paths["lock_path"], 'w', encoding='utf-8') as lock_file:
                lock_file.write(f"PID: {os.getpid()}, Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

            try:
                return self._download_and_save_model(model_id, model_version, paths)
            except (ConnectionError, TimeoutError, OSError) as e:
                logger.error("模型下载失败: %s", e)
                try:
                    incomplete_marker = os.path.join(paths["cache_path"], ".incomplete")
                    with open(incomplete_marker, 'w', encoding='utf-8') as f:
                        f.write(f"Download failed: {str(e)}")
                except (PermissionError, OSError) as mark_err:
                    logger.error("创建下载失败标记失败: %s", mark_err)
                raise RuntimeError(f"模型下载失败: {str(e)}") from e
            finally:
                try:
                    if os.path.exists(paths["lock_path"]):
                        os.remove(paths["lock_path"])
                except (PermissionError, OSError) as del_err:
                    logger.error("删除锁文件失败: %s", del_err)
        except (PermissionError, OSError) as lock_err:
            logger.error("创建锁文件失败: %s", lock_err)
            raise RuntimeError(f"无法创建模型下载锁，可能没有写入权限: {paths['lock_path']}") from lock_err
