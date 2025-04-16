"""
错误码定义模块，定义了各种操作可能返回的错误码
"""

# 通用错误码
SUCCESS = 0                    # 操作成功
UNKNOWN_ERROR = 1              # 未知错误
INVALID_ARGUMENT = 2           # 无效参数
NOT_INITIALIZED = 3            # 未初始化
ALREADY_INITIALIZED = 4        # 已经初始化
OPERATION_FAILED = 5           # 操作失败
NOT_IMPLEMENTED = 6            # 功能未实现
RESOURCE_UNAVAILABLE = 7       # 资源不可用
TIMEOUT = 8                    # 操作超时
RUNTIME_ERROR = 9              # 运行时错误
PERMISSION_DENIED = 10         # 权限被拒绝

# 模态管理相关错误码
MODALITY_NOT_FOUND = 100        # 未找到指定的模态
MODALITY_ALREADY_EXISTS = 101   # 模态已存在
MODALITY_REGISTRATION_FAILED = 102  # 模态注册失败
MODALITY_START_FAILED = 103     # 模态启动失败
MODALITY_STOP_FAILED = 104      # 模态停止失败

# 视觉模态相关错误码
VIDEO_SOURCE_ERROR = 200         # 视频源错误
CAMERA_NOT_AVAILABLE = 201       # 摄像头不可用
VIDEO_FILE_NOT_FOUND = 202       # 视频文件未找到
FRAME_ACQUISITION_FAILED = 203   # 帧获取失败
VIDEO_PROCESSING_ERROR = 204     # 视频处理错误

# 头部跟踪器相关错误码
FACE_DETECTION_ERROR = 300       # 人脸检测错误
HEAD_TRACKING_FAILED = 301       # 头部跟踪失败
MEDIAPIPE_INITIALIZATION_FAILED = 302  # MediaPipe初始化失败

# 错误码映射到描述信息
ERROR_DESCRIPTIONS = {
    # 通用错误码描述
    SUCCESS: "操作成功",
    UNKNOWN_ERROR: "未知错误",
    INVALID_ARGUMENT: "无效参数",
    NOT_INITIALIZED: "未初始化",
    ALREADY_INITIALIZED: "已经初始化",
    OPERATION_FAILED: "操作失败",
    NOT_IMPLEMENTED: "功能未实现",
    RESOURCE_UNAVAILABLE: "资源不可用",
    TIMEOUT: "操作超时",
    RUNTIME_ERROR: "运行时错误",
    PERMISSION_DENIED: "权限被拒绝",
    
    # 模态管理相关错误码描述
    MODALITY_NOT_FOUND: "未找到指定的模态",
    MODALITY_ALREADY_EXISTS: "模态已存在",
    MODALITY_REGISTRATION_FAILED: "模态注册失败",
    MODALITY_START_FAILED: "模态启动失败",
    MODALITY_STOP_FAILED: "模态停止失败",
    
    # 视觉模态相关错误码描述
    VIDEO_SOURCE_ERROR: "视频源错误",
    CAMERA_NOT_AVAILABLE: "摄像头不可用",
    VIDEO_FILE_NOT_FOUND: "视频文件未找到",
    FRAME_ACQUISITION_FAILED: "帧获取失败",
    VIDEO_PROCESSING_ERROR: "视频处理错误",
    
    # 头部跟踪器相关错误码描述
    FACE_DETECTION_ERROR: "人脸检测错误",
    HEAD_TRACKING_FAILED: "头部跟踪失败",
    MEDIAPIPE_INITIALIZATION_FAILED: "MediaPipe初始化失败"
}

def get_error_message(error_code: int) -> str:
    """
    获取错误码对应的描述信息
    
    Args:
        error_code: 错误码
        
    Returns:
        str: 错误描述信息，如果错误码未定义则返回"未定义的错误码"
    """
    return ERROR_DESCRIPTIONS.get(error_code, f"未定义的错误码: {error_code}")