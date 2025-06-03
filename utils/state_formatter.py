"""状态格式化器，用于将各模态状态转换为LLM可以理解的格式"""
import datetime
import json
from typing import Any, Dict, List


class StateFormatter:
    """
    状态格式化器，处理各种模态状态的规范化和优化，
    使语言模型更容易理解和处理
    """
    
    def __init__(self):
        """初始化状态格式化器"""
        pass
    
    def format_modality_states(self, states: Dict[str, Any]) -> Dict[str, Any]:
        """
        格式化所有模态状态
        
        Args:
            states: 模态状态字典，键为模态名称，值为状态对象
            
        Returns:
            格式化后的状态字典
        """
        formatted_states = {}
        
        for modality_name, state in states.items():
            if state is None:
                continue
                
            # 将模态状态转换为字典
            if hasattr(state, "to_dict") and callable(state.to_dict):
                state_dict = state.to_dict()
            else:
                state_dict = state
                
            # 根据不同的模态类型进行特定处理
            if "speech_recognition" in modality_name:
                formatted_states["speech"] = self._format_speech_state(state_dict)
            elif "head_pose" in modality_name:
                formatted_states["head_pose"] = self._format_head_pose_state(state_dict)
            elif "gaze" in modality_name:
                formatted_states["gaze"] = self._format_gaze_state(state_dict)
            elif "gesture" in modality_name:
                formatted_states["gesture"] = self._format_gesture_state(state_dict)
            else:
                # 对于未知模态，保留原始状态但移除帧数据
                if isinstance(state_dict, dict) and "frame" in state_dict:
                    state_dict.pop("frame", None)
                formatted_states[modality_name] = state_dict
        
        return formatted_states
    
    def _format_speech_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        格式化语音状态
        
        Args:
            state: 原始语音状态
            
        Returns:
            格式化后的语音状态
        """
        if "recognition" not in state:
            return state
            
        recognition = state["recognition"]
        
        # 提取关键信息
        formatted_speech = {
            "text": recognition.get("text", ""),
            "is_command": recognition.get("is_command", False),
            "has_wake_word": recognition.get("has_wake_word", False),
            "wake_word": recognition.get("wake_word", ""),
            "speaker_name": recognition.get("speaker_name", ""),
            "is_registered_speaker": recognition.get("is_registered_speaker", False),
            "confidence": recognition.get("confidence", 0.0)
        }
        
        # 只有在有关键词时才添加关键词信息
        if recognition.get("has_keyword", False):
            formatted_speech["keyword"] = recognition.get("keyword", "")
            formatted_speech["keyword_category"] = recognition.get("keyword_category", "")
        
        return formatted_speech
    
    def _format_head_pose_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        格式化头部姿态状态
        
        Args:
            state: 原始头部姿态状态
            
        Returns:
            格式化后的头部姿态状态
        """
        if "head state" not in state and "detections" not in state:
            return state
            
        head_state = state.get("head state", state.get("detections", {}))
        
        if not head_state:
            return {}
            
        head_pose = head_state.get("head_pose", {})
        head_movement = head_state.get("head_movement", {})
        
        # 提取关键信息
        formatted_head_pose = {
            "pose": {
                "pitch": head_pose.get("pitch", 0.0),  # 俯仰角（点头）
                "yaw": head_pose.get("yaw", 0.0),      # 偏航角（左右转头）
                "roll": head_pose.get("roll", 0.0),    # 翻滚角（头部倾斜）
                "detected": head_pose.get("detected", False)
            },
            "movement": {
                "is_nodding": head_movement.get("is_nodding", False),
                "is_shaking": head_movement.get("is_shaking", False),
                "status": head_movement.get("status", "stationary")
            }
        }
        
        return formatted_head_pose
    
    def _format_gaze_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        格式化视线方向状态
        
        Args:
            state: 原始视线方向状态
            
        Returns:
            格式化后的视线方向状态
        """
        # 处理不同的状态结构
        gaze_info = None
        
        if "gaze boost" in state:
            gaze_info = state["gaze boost"]
        elif "gaze_direction" in state.get("detections", {}):
            gaze_info = state["detections"]["gaze_direction"]
            
        if not gaze_info:
            return {}
            
        # 提取关键信息
        formatted_gaze = {
            "horizontal_direction": gaze_info.get("horizontal", {}).get("direction", "center"),
            "vertical_direction": gaze_info.get("vertical", {}).get("direction", "center"),
            "combined_direction": gaze_info.get("combined_direction", "center"),
            "face_detected": gaze_info.get("face_detected", False)
        }
        
        return formatted_gaze
    
    def _format_gesture_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        格式化手势状态
        
        Args:
            state: 原始手势状态
            
        Returns:
            格式化后的手势状态
        """
        # 处理不同的状态结构
        gesture_info = None
        
        if "hand state" in state:
            gesture_info = state["hand state"]
        elif "gesture" in state.get("detections", {}):
            gesture_info = state["detections"]["gesture"]
            
        if not gesture_info:
            return {}
            
        # 提取关键信息
        formatted_gesture = {
            "id": gesture_info.get("id", 10),  # 默认为"ignore" (10)
            "name": gesture_info.get("name", "ignore"),
            "confidence": gesture_info.get("confidence", 0.0),
            "detected": gesture_info.get("detected", False),
            "stability": gesture_info.get("stability", 0.0)
        }
        
        return formatted_gesture
    
    def add_context_info(self, formatted_states: Dict[str, Any]) -> Dict[str, Any]:
        """
        添加上下文信息，如时间、日期等
        
        Args:
            formatted_states: 格式化后的状态字典
            
        Returns:
            添加了上下文信息的状态字典
        """
        now = datetime.datetime.now()
        
        context_info = {
            "timestamp": now.timestamp(),
            "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
            "weekday": now.strftime("%A")
        }
        
        formatted_states["context"] = context_info
        return formatted_states
    
    def process(self, states: Dict[str, Any]) -> str:
        """
        处理所有模态状态并生成LLM友好的JSON字符串
        
        Args:
            states: 原始模态状态字典
            
        Returns:
            JSON字符串表示的格式化状态
        """
        # 格式化模态状态
        formatted_states = self.format_modality_states(states)
        
        # 添加上下文信息
        formatted_states_with_context = self.add_context_info(formatted_states)
        
        # 转换为JSON字符串
        return json.dumps(formatted_states_with_context, ensure_ascii=False, indent=2)