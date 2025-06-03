import requests
import json
import os
import time
import datetime
from pathlib import Path

class ConversationManager:
    def __init__(self, api_key, model="Qwen/Qwen3-30B-A3B", conversation_id=None, max_history_length=20):
        """
        初始化对话管理器
        
        参数:
            api_key (str): API密钥
            model (str): 使用的模型名称
            conversation_id (str): 对话ID，如果为None则自动生成
            max_history_length (int): 保留的最大历史消息数量
        """
        self.api_key = api_key
        self.model = model
        self.conversation_history = []
        self.tool_call_history = []
        self.max_history_length = max_history_length
        self.url = "https://api.siliconflow.cn/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # 设置对话ID和持久化存储路径
        self.conversation_id = conversation_id or f"conversation_{int(time.time())}"
        self._setup_storage_paths()
    
    def _setup_storage_paths(self):
        """设置持久化存储路径"""
        # 获取项目根目录路径
        root_dir = Path(__file__).parent.parent.parent
        
        # 创建对话历史存储目录
        self.storage_dir = root_dir / "database" / "conversation"
        if not self.storage_dir.exists():
            os.makedirs(self.storage_dir)
            
        # 设置对话历史文件路径
        self.history_file = self.storage_dir / f"{self.conversation_id}.json"
        self.tool_calls_file = self.storage_dir / f"{self.conversation_id}_tools.json"
        
        # 如果存储文件已存在，则加载历史记录
        self._load_history()
    
    def _load_history(self):
        """从文件加载历史记录"""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    self.conversation_history = json.load(f)
            except Exception as e:
                print(f"加载对话历史失败: {e}")
        
        if self.tool_calls_file.exists():
            try:
                with open(self.tool_calls_file, 'r', encoding='utf-8') as f:
                    self.tool_call_history = json.load(f)
            except Exception as e:
                print(f"加载工具调用历史失败: {e}")
    
    def _save_history(self):
        """将历史记录保存到文件"""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.conversation_history, f, ensure_ascii=False, indent=2)
            
            with open(self.tool_calls_file, 'w', encoding='utf-8') as f:
                json.dump(self.tool_call_history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存对话历史失败: {e}")
    
    def _manage_history_length(self):
        """控制历史记录长度，保留最近的n条消息"""
        if len(self.conversation_history) > self.max_history_length:
            # 始终保留第一条系统消息（如果存在）
            if self.conversation_history and self.conversation_history[0].get("role") == "system":
                self.conversation_history = [self.conversation_history[0]] + self.conversation_history[-(self.max_history_length-1):]
            else:
                self.conversation_history = self.conversation_history[-self.max_history_length:]
    
    def add_user_message(self, message):
        """添加用户消息到对话历史"""
        msg = {"role": "user", "content": message, "timestamp": datetime.datetime.now().isoformat()}
        self.conversation_history.append({"role": "user", "content": message})
        self._manage_history_length()
        self._save_history()
    
    def add_system_message(self, message):
        """添加系统消息到对话历史"""
        msg = {"role": "system", "content": message, "timestamp": datetime.datetime.now().isoformat()}
        self.conversation_history.append({"role": "system", "content": message})
        self._manage_history_length()
        self._save_history()
    
    def add_tool_result(self, tool_call_id, content):
        """添加工具调用结果到对话历史"""
        msg = {"role": "tool", "tool_call_id": tool_call_id, "content": content}
        self.conversation_history.append(msg)
        self._manage_history_length()
        self._save_history()
    
    def clear_history(self):
        """清空对话历史"""
        self.conversation_history = []
        self._save_history()
    
    def get_last_n_messages(self, n):
        """获取最近的n条消息"""
        return self.conversation_history[-n:] if n < len(self.conversation_history) else self.conversation_history
    
    def record_tool_call(self, tool_call):
        """记录工具调用"""
        tool_record = {
            "timestamp": datetime.datetime.now().isoformat(),
            "tool_call": tool_call
        }
        self.tool_call_history.append(tool_record)
        self._save_history()
        return tool_record
    
    def get_tool_call_history(self):
        """获取工具调用历史"""
        return self.tool_call_history
    
    def get_response(self, max_tokens=512, temperature=0.7, tools=None):
        """发送请求并获取回复"""
        # 准备请求负载
        payload = {
            "model": self.model,
            "messages": self.conversation_history,
            "stream": False,
            "max_tokens": max_tokens,
            "enable_thinking": False,
            "thinking_budget": 4096,
            "temperature": temperature,
            "top_p": 0.7,
            "top_k": 50,
            "frequency_penalty": 0.5,
        }
        
        if tools:
            payload["tools"] = tools
        
        # 发送请求
        try:
            start_time = time.time()
            response = requests.request("POST", self.url, json=payload, headers=self.headers)
            end_time = time.time()
            
            response_data = response.json()
            
            # 请求失败处理
            if "error" in response_data:
                error_msg = {"error": response_data.get("error", "Unknown error"), "raw_response": response_data}
                return error_msg
                
            # 处理响应结果
            if "choices" in response_data and len(response_data["choices"]) > 0:
                assistant_message = response_data["choices"][0]["message"]
                
                # 将响应时间添加到消息中（仅用于记录，不发送给API）
                assistant_message_with_meta = assistant_message.copy()
                assistant_message_with_meta["response_time"] = end_time - start_time
                assistant_message_with_meta["timestamp"] = datetime.datetime.now().isoformat()
                
                # 将助手回复添加到对话历史（不包含元数据）
                self.conversation_history.append(assistant_message)
                self._manage_history_length()
                self._save_history()
                
                # 处理工具调用
                if "tool_calls" in assistant_message:
                    # 记录工具调用
                    for tool_call in assistant_message["tool_calls"]:
                        self.record_tool_call(tool_call)
                
                return assistant_message
            
            return {"error": "No valid response", "raw_response": response_data}
            
        except Exception as e:
            return {"error": str(e)}

# 导出单例实例
_conversation_manager = None

def get_conversation_manager(api_key=None, model=None):
    """获取全局对话管理器实例"""
    global _conversation_manager
    if _conversation_manager is None and api_key:
        _conversation_manager = ConversationManager(api_key=api_key, model=model)
        _conversation_manager.add_system_message("""
你是一个智能车载助手，需要根据多模态信息判断驾驶员状态和意图，并采取适当行动。

你会接收以下类型的驾驶员信息：
1. 视线朝向：包含"center"(中间)、"left"(左)、"right"(右)、"up"(上)、"down"(下)等方向
2. 头部姿态：包含"nodding"(点头)、"shaking"(摇头)、"stationary"(静止)等状态
3. 手势动作：如"握拳"(0)、"摇手"(5)、"竖起大拇指"(6)等手势
4. 语音输入：驾驶员的语音指令

分析规则：
- 当驾驶员视线离开前方时，可能需要提醒注意安全
- 当检测到点头动作时，通常表示确认或同意
- 当检测到摇头动作时，通常表示否定或拒绝
- 特定手势可触发对应功能，如"竖起大拇指"表示赞同
- 结合上下文理解语音指令，优先级高于肢体语言

你可以执行的操作包括但不限于：
- 提醒驾驶员注意路况
- 调整车内环境（温度、音乐等）
- 导航和路线规划
- 回答驾驶员问询
- 紧急情况处理（如检测到驾驶员疲劳）
                                                 
但目前暂时只实现了语音工具的接入，因此当你认为你需要做什么的适合，暂时通过语音工具来返回。

请根据接收到的多模态信息，理解驾驶员的真实需求，并通过调用合适的工具来提供精准、及时的帮助。""")
    return _conversation_manager