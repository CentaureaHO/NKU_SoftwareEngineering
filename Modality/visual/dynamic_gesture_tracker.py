import cv2
import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Union
import time
import logging
import os
import torch
from collections import deque
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import Compose, CenterCrop, ToTensor, Normalize
import json

# 配置日志
logging.basicConfig(
    level=logging.DEBUG if os.environ.get('MODALITY_DEBUG', '0') == '1' else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='dynamic_gesture_tracker.log',
    filemode='w'
)
logger = logging.getLogger('DynamicGestureTracker')

DEBUG = os.environ.get('MODALITY_DEBUG', '0') == '1'

# 导入基类和错误码
from .base_visual import BaseVisualModality, VisualState
from ..core.error_codes import (
    SUCCESS, CAMERA_NOT_AVAILABLE, VIDEO_FILE_NOT_FOUND, 
    VIDEO_SOURCE_ERROR, FRAME_ACQUISITION_FAILED, RUNTIME_ERROR,
    MODEL_LOADING_FAILED, MODEL_NOT_FOUND
)

# 设备选择
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DynamicGestureState(VisualState):
    """动态手势状态类"""
    
    def __init__(self, frame=None, timestamp=None):
        super().__init__(frame, timestamp)
        self.detections = {
            "gesture": {
                "id": -1,
                "name": "等待手势...",
                "confidence": 0.0,
                "detected": False,
                "motion_level": 0.0,
                "stability": 0.0
            }
        }
    
    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        return result

def detect_motion(frames, threshold=500):
    """检测帧序列中是否有足够的运动
    
    参数:
    frames -- 帧列表，每个元素是一个张量
    threshold -- 运动检测阈值，值越大需要的运动越明显
    
    返回值:
    motion_detected -- 布尔值，表示是否检测到足够的运动
    motion_level -- 运动量值，用于调试
    """
    if len(frames) < 2:
        return False, 0
    
    # 转换为numpy数组以便计算
    np_frames = []
    for frame in frames:
        # 将张量转换为numpy数组
        np_frame = frame.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        # 转换为灰度图
        np_frame = np.mean(np_frame, axis=2)
        np_frames.append(np_frame)
    
    # 计算帧间差异总和
    total_diff = 0
    for i in range(1, len(np_frames)):
        diff = np.abs(np_frames[i] - np_frames[i-1])
        total_diff += np.sum(diff)
    
    # 根据阈值判断是否有运动
    motion_level = total_diff / (len(np_frames) - 1)
    motion_detected = motion_level > threshold
    
    return motion_detected, motion_level

def load_labels(label_path):
    """加载标签文件"""
    labels = []
    with open(label_path) as f:
        for line in f:
            labels.append(line.strip())
    return labels

def load_model(model_path, model_class_path, num_classes=None):
    """加载训练好的模型"""
    try:
        # 如果未指定类别数，尝试从模型检测
        if num_classes is None:
            # 尝试从checkpoint获取类别数
            checkpoint = torch.load(model_path, map_location=DEVICE)
            
            # 尝试不同的方法获取类别数
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                # 检查最后一层全连接层的参数形状
                for key in state_dict:
                    if 'fc6.weight' in key:
                        num_classes = state_dict[key].size(0)
                        break
                    if 'fc.weight' in key:
                        num_classes = state_dict[key].size(0)
                        break
            
            if num_classes is None:
                num_classes = 27  # 默认为Jester数据集的27个类别
        
        # 动态导入模型类
        import importlib.util
        import sys
        
        # 获取模型文件的绝对路径
        model_file_path = os.path.abspath(model_class_path)
        module_name = os.path.basename(model_class_path).replace('.py', '')
        
        # 加载模块
        spec = importlib.util.spec_from_file_location(module_name, model_file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        
        # 创建模型实例
        model_class = getattr(module, 'ConvColumn')
        model = model_class(num_classes)
        
        # 加载权重
        checkpoint = torch.load(model_path, map_location=DEVICE)
        state_dict = checkpoint['state_dict']
        
        # 处理模型权重中的DataParallel前缀
        if list(state_dict.keys())[0].startswith('module.'):
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # 去掉'module.'前缀
                new_state_dict[name] = v
            state_dict = new_state_dict
        
        model.load_state_dict(state_dict)
        model = model.to(DEVICE)
        model.eval()
        return model, num_classes
    
    except Exception as e:
        logger.error(f"加载模型时出错: {e}")
        return None, 0

def preprocess_frame(frame, transform):
    """预处理单帧图像，确保与训练时处理一致"""
    # 将OpenCV的BGR转换为RGB
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    # 直接应用transform
    img = transform(img)
    return img

def put_chinese_text(img, text, position, font_size=20, color=(255, 255, 255), thickness=1):
    """在OpenCV图像上绘制中文文本"""
    # 创建一个空白的PIL图像，绘制文本，然后转换回OpenCV格式
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    # 尝试加载系统字体，如果找不到，则使用默认的等宽字体
    try:
        # 优先尝试微软雅黑或其他常见中文字体
        font_paths = [
            "C:/Windows/Fonts/msyh.ttc",         # Windows 微软雅黑
            "C:/Windows/Fonts/simhei.ttf",       # Windows 黑体
            "C:/Windows/Fonts/simsun.ttc",       # Windows 宋体
            "/System/Library/Fonts/PingFang.ttc" # macOS
        ]
        
        font = None
        for path in font_paths:
            if os.path.exists(path):
                font = ImageFont.truetype(path, font_size)
                break
                
        if font is None:
            # 如果找不到上述字体，使用默认字体
            font = ImageFont.load_default()
    except IOError:
        # 如果加载失败，使用默认字体
        font = ImageFont.load_default()
    
    # 在PIL图像上绘制文本
    draw = ImageDraw.Draw(img_pil)
    draw.text(position, text, font=font, fill=color)
    
    # 转换回OpenCV格式
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return img_cv

class DynamicGestureTracker(BaseVisualModality):
    """
    动态手势跟踪器，识别用户的动态手势
    """
    
    def __init__(self, name: str = "dynamic_gesture_tracker", source: int = 0, 
                 width: int = 640, height: int = 480,
                 model_path: str = 'Modality/models/dynamic_gesture_recognition/jester_conv_example/model_best.pth.tar',
                 model_class_path: str = 'Modality/models/dynamic_gesture_recognition/model.py',
                 label_path: str = 'Dataset/dynamic_gesture_recognition/annotations/jester-v1-labels.csv',
                 confidence_threshold: float = 0.65,
                 motion_threshold: float = 500.0,
                 clip_size: int = 16,
                 history_size: int = 6,
                 min_consensus: int = 4,
                 prediction_cooldown: float = 1.0,
                 input_width: int = 176,
                 input_height: int = 100,
                 debug: bool = DEBUG):
        """
        初始化动态手势跟踪器
        
        Args:
            name: 模态名称
            source: 视频源，可以是摄像头ID或视频文件路径
            width: 图像宽度
            height: 图像高度
            model_path: 模型路径
            model_class_path: 模型类文件路径
            label_path: 标签文件路径
            confidence_threshold: 置信度阈值
            motion_threshold: 运动检测阈值
            clip_size: 视频片段长度
            history_size: 历史记录大小
            min_consensus: 达成共识所需的最小样本数
            prediction_cooldown: 预测之间的冷却时间(秒)
            input_width: 输入到模型前的图像宽度
            input_height: 输入到模型前的图像高度
            debug: 是否启用调试模式
        """
        super().__init__(name, source, width, height)
        self.model_path = model_path
        self.model_class_path = model_class_path
        self.label_path = label_path
        self.confidence_threshold = confidence_threshold
        self.motion_threshold = motion_threshold
        self.clip_size = clip_size
        self.history_size = history_size
        self.min_consensus = min_consensus
        self.prediction_cooldown = prediction_cooldown
        self.input_width = input_width
        self.input_height = input_height
        self.debug = debug
        
        # 初始化状态
        self.model = None
        self.loaded = False
        self.labels = []
        self.num_classes = 0
        
        # 帧缓冲区和历史记录
        self.frames_buffer = []
        self.prediction_history = deque(maxlen=self.history_size)
        self.last_prediction_time = 0
        self.current_prediction = "等待手势..."
        self.current_confidence = 0.0
        self.current_motion_level = 0.0
        
        # 变换
        self.transform = Compose([
            CenterCrop(84),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225])
        ])
        
        logger.info(f"动态手势跟踪器初始化完成，调试模式：{self.debug}")
    
    def set_parameters(self, history_size=None, min_consensus=None, 
                     prediction_cooldown=None, motion_threshold=None):
        """
        动态设置参数
        
        Args:
            history_size: 历史记录大小
            min_consensus: 达成共识所需的最小样本数
            prediction_cooldown: 预测之间的冷却时间(秒)
            motion_threshold: 运动检测阈值
        """
        if history_size is not None:
            self.history_size = history_size
            # 重新创建历史记录队列
            old_history = list(self.prediction_history)
            self.prediction_history = deque(maxlen=self.history_size)
            # 保留最近的历史记录
            for item in old_history[-self.history_size:]:
                self.prediction_history.append(item)
        
        if min_consensus is not None:
            self.min_consensus = min_consensus
        
        if prediction_cooldown is not None:
            self.prediction_cooldown = prediction_cooldown
        
        if motion_threshold is not None:
            self.motion_threshold = motion_threshold
        
        logger.info(f"更新参数: 历史大小={self.history_size}, 共识={self.min_consensus}, " +
                   f"冷却时间={self.prediction_cooldown}s, 运动阈值={self.motion_threshold}")
        
        return True
    
    def initialize(self) -> int:
        """
        初始化摄像头和模型
        
        Returns:
            int: 错误码，0表示成功，其他值表示失败
        """
        result = super().initialize()
        if result != SUCCESS:
            logger.error("基础视觉模态初始化失败")
            return result
        
        try:
            # 检查并加载标签
            if not os.path.exists(self.label_path):
                logger.error(f"错误: 标签文件不存在 {self.label_path}")
                return MODEL_NOT_FOUND
            
            self.labels = load_labels(self.label_path)
            logger.info(f"已加载标签 ({len(self.labels)}): {self.labels}")
            
            # 检查模型文件
            if not os.path.exists(self.model_path):
                logger.error(f"错误: 模型文件不存在 {self.model_path}")
                return MODEL_NOT_FOUND
            
            # 检查模型类文件
            if not os.path.exists(self.model_class_path):
                logger.error(f"错误: 模型类文件不存在 {self.model_class_path}")
                return MODEL_NOT_FOUND
            
            # 加载模型
            self.model, self.num_classes = load_model(self.model_path, self.model_class_path)
            if self.model is None:
                logger.error("模型加载失败")
                return MODEL_LOADING_FAILED
            
            logger.info(f"成功加载模型: {self.model_path}, 类别数: {self.num_classes}")
            self.loaded = True
            
            return SUCCESS
        except Exception as e:
            logger.error(f"初始化时出错: {str(e)}")
            return RUNTIME_ERROR
    
    def _process_frame(self, frame: np.ndarray) -> DynamicGestureState:
        """
        处理图像帧，识别动态手势
        
        Args:
            frame: 输入图像帧
            
        Returns:
            DynamicGestureState: 动态手势状态
        """
        state = DynamicGestureState(frame=frame)
        
        if not self.loaded:
            return state
        
        try:
            # 首先将整个帧缩放到指定的低分辨率，然后再缩放到模型所需的输入尺寸
            frame_small = cv2.resize(frame, (self.input_width, self.input_height))
            frame_for_model = cv2.resize(frame_small, (224, 224))
            
            # 预处理并添加到缓冲区
            img_tensor = preprocess_frame(frame_for_model, self.transform)
            self.frames_buffer.append(img_tensor.unsqueeze(0))
            
            # 保持缓冲区大小
            if len(self.frames_buffer) > self.clip_size:
                self.frames_buffer.pop(0)
            
            # 检查是否已经过了冷却时间
            current_time = time.time()
            cooldown_elapsed = current_time - self.last_prediction_time >= self.prediction_cooldown
            
            # 当有足够的帧时，先检测运动，再进行预测
            if len(self.frames_buffer) == self.clip_size and cooldown_elapsed:
                # 检测运动
                motion_detected, motion_level = detect_motion(self.frames_buffer, threshold=self.motion_threshold)
                self.current_motion_level = motion_level
                
                # 更新状态中的运动级别
                state.detections["gesture"]["motion_level"] = float(motion_level)
                
                # 在debug模式下，无论是否检测到足够的运动，都计算并显示前五个预测结果
                if self.debug:
                    # 准备输入数据计算前五个预测结果
                    input_data = torch.cat(self.frames_buffer, dim=0).unsqueeze(0)  # [1, T, C, H, W]
                    # 调整维度顺序：从[1, T, C, H, W]到[1, C, T, H, W]
                    input_data = input_data.permute(0, 2, 1, 3, 4)
                    input_data = input_data.to(DEVICE)
                    
                    # 预测
                    with torch.no_grad():
                        outputs = self.model(input_data)
                        probs = torch.nn.functional.softmax(outputs, dim=1)
                        
                        # 获取前5个预测
                        values, indices = torch.topk(probs[0], min(5, probs.size(1)), 0)
                        top5_predictions = []
                        for i in range(min(5, len(self.labels))):
                            idx = indices[i].item()
                            if idx < len(self.labels):
                                label = self.labels[idx]
                                conf = values[i].item()
                                top5_predictions.append((label, float(conf)))
                        state.detections["gesture"]["top_predictions"] = top5_predictions
                
                # 只有检测到足够的运动时才进行手势识别及其后续处理
                if motion_detected:
                    # 准备输入数据
                    input_data = torch.cat(self.frames_buffer, dim=0).unsqueeze(0)  # [1, T, C, H, W]
                    # 调整维度顺序：从[1, T, C, H, W]到[1, C, T, H, W]
                    input_data = input_data.permute(0, 2, 1, 3, 4)
                    input_data = input_data.to(DEVICE)
                    
                    # 预测
                    with torch.no_grad():
                        outputs = self.model(input_data)
                        probs = torch.nn.functional.softmax(outputs, dim=1)
                        
                        # 获取最佳预测
                        confidence, prediction = torch.max(probs, 1)
                        conf_value = confidence.item()
                        
                        # 如果置信度超过阈值，记录预测
                        if conf_value > self.confidence_threshold and prediction.item() < len(self.labels):
                            pred_label = self.labels[prediction.item()]
                            
                            # 添加到历史记录
                            self.prediction_history.append(pred_label)
                            
                            # 统计历史记录中的频率
                            from collections import Counter
                            prediction_counts = Counter(self.prediction_history)
                            most_common = prediction_counts.most_common(1)
                            
                            # 只有当某个手势在历史记录中出现足够次数时才更新显示结果
                            if most_common and most_common[0][1] >= self.min_consensus:
                                stable_prediction = most_common[0][0]
                                # 如果预测结果已经改变，更新显示结果
                                if stable_prediction != self.current_prediction:
                                    self.current_prediction = stable_prediction
                                    self.current_confidence = conf_value
                                    self.last_prediction_time = current_time  # 更新预测时间
                                    
                                    # 找到手势ID
                                    try:
                                        gesture_id = self.labels.index(stable_prediction)
                                    except ValueError:
                                        gesture_id = -1
                                    
                                    # 计算稳定性
                                    stability = most_common[0][1] / len(self.prediction_history)
                                    
                                    # 更新状态
                                    state.detections["gesture"]["id"] = gesture_id
                                    state.detections["gesture"]["name"] = stable_prediction
                                    state.detections["gesture"]["confidence"] = float(conf_value)
                                    state.detections["gesture"]["detected"] = True
                                    state.detections["gesture"]["stability"] = float(stability)
                                    
                                    if self.debug:
                                        logger.debug(f"检测到手势: {stable_prediction}, 置信度: {conf_value:.2f}, 稳定性: {stability:.2f}")
                else:
                    # 未检测到足够的运动
                    if self.debug:
                        logger.debug(f"未检测到足够的运动: {motion_level:.1f} < {self.motion_threshold}")
            
            # 如果当前有活动的预测，更新状态
            if self.current_prediction != "等待手势..." and state.detections["gesture"]["name"] == "等待手势...":
                # 找到手势ID
                try:
                    gesture_id = self.labels.index(self.current_prediction)
                except ValueError:
                    gesture_id = -1
                
                # 更新状态
                state.detections["gesture"]["id"] = gesture_id
                state.detections["gesture"]["name"] = self.current_prediction
                state.detections["gesture"]["confidence"] = float(self.current_confidence)
                state.detections["gesture"]["detected"] = True
            
        except Exception as e:
            logger.error(f"处理帧时出错: {str(e)}")
        
        return state
    
    def shutdown(self) -> int:
        """
        关闭动态手势跟踪
        
        Returns:
            int: 错误码，0表示成功，其他值表示失败
        """
        try:
            # 清空帧缓冲区
            self.frames_buffer.clear()
            
            # 释放模型
            self.model = None
            self.loaded = False
            
            # 调用父类方法关闭视频源
            result = super().shutdown()
            
            logger.info("动态手势跟踪器已关闭")
            return result
        except Exception as e:
            logger.error(f"关闭动态手势跟踪器时出错: {str(e)}")
            return RUNTIME_ERROR 