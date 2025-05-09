#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#_author = 'Yidian Lin'

"""
Module Description:
    用于实现车载多模态智能交互系统的控制功能
"""

import os
import time
import argparse
from Modality.core.modality_manager import ModalityManager
from Modality.speech.speech_recognition import SpeechRecognition
from Modality.core.error_codes import SUCCESS, get_error_message
import cv2
import time
import argparse
import os
import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
from Modality.core import ModalityManager
from Modality.core.error_codes import SUCCESS
import cv2
import time
import os
import numpy as np
import sys
import argparse
from PIL import Image, ImageDraw, ImageFont
# 确保可以导入Modality模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from individuation import Individuation
from Modality import ModalityManager, GestureTracker
from viewer.viewer import init_viewer
import threading
import webbrowser

class MultimodalController:
    # TODO():这个函数负责启动时打开各个模态,目前还未实现视线跟踪模态
    def __init__(self) -> None:
        # 启动语音模态
        self.manager = ModalityManager()
        self.init_speecher()
        self.init_headpose()
        self.init_static_gesture()
        self.init_ui()

    def init_ui(self) -> None:
        flask_thread = threading.Thread(target=init_viewer)
        flask_thread.daemon = True  # 设置为守护线程，主线程退出时自动结束
        flask_thread.start()
        webbrowser.open("http://127.0.0.1:5000")

    def init_speecher(self) -> None:
        parser = argparse.ArgumentParser(description='智能座舱语音识别演示')
        parser.add_argument('--no-wake', action='store_true', help='关闭唤醒词功能')
        parser.add_argument('--no-sv', action='store_true', help='关闭声纹识别功能')
        parser.add_argument('--add-wake', type=str, help='添加自定义唤醒词')
        parser.add_argument('--register', action='store_true', help='强制进入声纹注册模式')
        parser.add_argument('--register-name', type=str, help='注册声纹的用户名')
        parser.add_argument('--max-temp', type=int, help='设置最大临时声纹数量', default=10)
        parser.add_argument('--debug', action='store_true', help='开启调试模式')
        args = parser.parse_args()
        
        if args.debug:
            os.environ["MODALITY_DEBUG"] = "1"
        else:
            os.environ["MODALITY_DEBUG"] = "0"
                
        print("正在初始化智能座舱语音识别系统...")
        self.speech_modality = SpeechRecognition(name="speech_recognition")
        print("语音识别模态创建成功")
        
        result = self.manager.register_modality(self.speech_modality)
        if result != SUCCESS:
            return
        
        print("语音识别模态注册成功")
        
        result = self.manager.start_modality("speech_recognition")
        if result != SUCCESS:
            return
        
        print("语音识别模态启动成功")

        if args.no_wake:
            self.speech_modality.toggle_wake_word(False)
        
        if args.add_wake:
            self.speech_modality.add_wake_word(args.add_wake)
        
        self.speech_modality.set_max_temp_speakers(args.max_temp)

        if args.register:
            name = args.register_name if args.register_name else "新用户"
            self.speech_modality.register_speaker(name)
        
        self.last_recognized_text = ""
        print("语音系统初始化完成")

    def init_headpose(self) -> None:
        parser = argparse.ArgumentParser(description="驾驶员监测系统演示")
        parser.add_argument("--camera", type=int, default=0, help="摄像头ID (默认为0)")
        parser.add_argument("--width", type=int, default=640, help="图像宽度 (默认为640)")
        parser.add_argument("--height", type=int, default=480, help="图像高度 (默认为480)")
        parser.add_argument("--video", type=str, default="", help="使用视频文件而不是摄像头")
        parser.add_argument("--record", type=str, default="", help="录制结果到视频文件")
        parser.add_argument("--debug", action="store_true", help="开启调试模式，显示更详细信息")
        parser.add_argument("--method", type=str, default="gru", choices=["geom", "gru"], help="选择方法 (默认为gru)")
        args = parser.parse_args()
    
        if args.debug:
            os.environ["MODALITY_DEBUG"] = "1"
        else:
            os.environ["MODALITY_DEBUG"] = "0"
        
        video_source = args.camera

        if args.method == "gru":
            from Modality.visual import HeadPoseTrackerGRU as Tracker
            print("使用 GRU 模型进行头部姿态检测")
        elif args.method == "geom":
            from Modality.visual import HeadPoseTrackerGeom as Tracker
            print("使用几何方法进行头部姿态检测")
        
        monitor = Tracker(
            source=video_source,
            width=args.width,
            height=args.height,
            debug=args.debug
        )
        
        result = self.manager.register_modality(monitor)
        if result != SUCCESS:
            return
        
        result = self.manager.start_modality(monitor.name)
        if result != SUCCESS:
            return
        
        face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        video_writer = None
        if args.record:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(args.record, fourcc, 20.0, (args.width, args.height))
        
        print("头部姿态识别系统初始化完成")

    def init_static_gesture(self) -> None:
        # 解析命令行参数
        parser = argparse.ArgumentParser(description='手势识别模态演示')
    
        # 摄像头相关参数
        parser.add_argument('--camera', type=int, default=0, help='摄像头设备ID (默认: 0)')
        parser.add_argument('--width', type=int, default=640, help='摄像头图像宽度 (默认: 640)')
        parser.add_argument('--height', type=int, default=480, help='摄像头图像高度 (默认: 480)')
        
        # 模型相关参数
        parser.add_argument('--model', type=str, default='Modality/models/static_gesture_recognition/model_output/gesture_model.h5', help='手势识别模型路径 (默认: Modality/models/static_gesture_recognition/model_output/gesture_model.h5)')
        parser.add_argument('--feature-mean', type=str, default='Modality/models/static_gesture_recognition/model_data/feature_mean.npy', help='特征均值文件路径 (默认: Modality/models/static_gesture_recognition/model_data/feature_mean.npy)')
        parser.add_argument('--feature-scale', type=str, default='Modality/models/static_gesture_recognition/model_data/feature_scale.npy', help='特征缩放文件路径 (默认: Modality/models/static_gesture_recognition/model_data/feature_scale.npy)')
        
        # 识别相关参数
        parser.add_argument('--confidence', type=float, default=0.75, help='手势识别置信度阈值 (默认: 0.75)')
        parser.add_argument('--stability', type=float, default=0.7, help='手势稳定性阈值 (默认: 0.7)')
        
        # 调试相关参数
        parser.add_argument('--debug', action='store_true', help='开启调试模式')
        args = parser.parse_args()
        
        # 如果模型目录不存在，创建它
        model_dir = os.path.dirname(args.model)
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)
        
        # 检查模型文件是否存在
        if not os.path.exists(args.model) or not os.path.exists(args.feature_mean) or not os.path.exists(args.feature_scale):
            print(f"错误: 模型文件不存在。请确保以下文件存在:")
            print(f"  - {args.model}")
            print(f"  - {args.feature_mean}")
            print(f"  - {args.feature_scale}")
            print("您可以通过命令行参数指定正确的模型文件路径:")
            print("  --model 模型文件路径")
            print("  --feature-mean 特征均值文件路径")
            print("  --feature-scale 特征缩放文件路径")
            return
        
        # 设置环境变量
        if args.debug:
            os.environ['MODALITY_DEBUG'] = '1'
        
        print(f"初始化手势识别模态...")
        print(f"  - 摄像头: {args.camera}")
        print(f"  - 分辨率: {args.width}x{args.height}")
        print(f"  - 模型: {args.model}")
        print(f"  - 置信度阈值: {args.confidence}")
        print(f"  - 稳定性阈值: {args.stability}")
        
        # 创建模态管理器
        manager = ModalityManager()
        
        # 创建手势识别模态
        static_gesture_tracker = GestureTracker(
            name="static_gesture_tracker",
            source=args.camera,
            width=args.width,
            height=args.height,
            model_path=args.model,
            feature_mean_path=args.feature_mean,
            feature_scale_path=args.feature_scale,
            confidence_threshold=args.confidence,
            stability_threshold=args.stability,
            debug=args.debug
        )
        
        # 注册模态
        self.manager.register_modality(static_gesture_tracker)
        
        # 启动模态
        result = self.manager.start_modality("static_gesture_tracker")
        if result != 0:
            print(f"错误: 启动手势识别模态失败，错误码: {result}")
            return
        
        self.last_gesture_name = None
        print("手势识别模态启动成功")

    def control_speecher(self,state) -> None:
        if state and state.recognition["text"]:
            current_text = state.recognition["text"]

            if current_text != self.last_recognized_text:
                self.last_recognized_text = current_text
                        
                result_text = []
                if state.recognition["has_wake_word"]:
                    result_text.append(f"[唤醒词: {state.recognition['wake_word']}]")
                if state.recognition["has_keyword"]:
                    result_text.append(f"[关键词: {state.recognition['keyword']} 类别: {state.recognition['keyword_category']}]")
                if state.recognition["speaker_id"]:
                    result_text.append(f"[说话人: {state.recognition['speaker_name']}]")
                        
                result_header = " ".join(result_text)
                if result_header:
                    print(f"\n{result_header}")
                print(f"识别结果: {current_text}")
                from individuation import Individuation
                Individuation.speech_individuation(current_text)


    def control_headpose(self,state) -> None:
        #from individuation import Individuation
        if state.detections['head_movement']['is_nodding']:
            print("点头")
            #Individuation.head_individuation("点头")
            time.sleep(0.5)
        elif state.detections['head_movement']['is_shaking']:
            print("摇头")
            #Individuation.head_individuation("摇头")
            time.sleep(0.5)

    def control_static_gesture(self,state) -> None:
        if state.detections["gesture"]["detected"]:
            name = state.detections["gesture"]["name"]
            if name == self.last_gesture_name:
                return
            self.last_gesture_name = name
            print(f"手势: {name}")
            Individuation.gesture_individuation(name)
            time.sleep(0.5)
    
    def control(self) -> None:
        try:
            while True:
                states = self.manager.update_all()
                for name, state in states.items():
                    # print(f"模态: {name}")
                    if name == "speech_recognition":
                        self.control_speecher(state)
                    elif name == "head_pose_tracker_gru":
                        self.control_headpose(state)
                    elif name == "static_gesture_tracker":
                        self.control_static_gesture(state)
                    else:
                        assert False, f"未知模态: {name}"
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n检测到终止信号")
        finally:
            print("正在关闭系统...")
            self.manager.shutdown_all()
            print("系统已关闭")

if __name__ == '__main__':
    controller = MultimodalController()
    controller.control()