#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# _author = 'Yidian Lin','Xianda Tang',"Yu Jiang"

"""
Module Description:
    用于实现车载多模态智能交互系统的控制功能
"""

import argparse
import os
import sys
import time

import mediapipe as mp

from logger import logger
from modality import GestureTracker, ModalityManager
from modality.core.error_codes import SUCCESS
from modality.speech.speech_recognition import SpeechRecognition
from modality.visual import GazeDirectionTracker, HeadPoseTrackerGRU
from components import get_component
from utils.tools import speecher_player

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


class MultimodalController:
    "多模态控制类"

    def __init__(self) -> None:
        "构造函数"
        self.manager = ModalityManager()
        self.init_speecher()
        self.init_headpose()
        self.init_static_gesture()
        self.init_gazer()

    def init_speecher(self) -> None:
        "初始化语音模态"
        parser = argparse.ArgumentParser(description="智能座舱语音识别演示")
        parser.add_argument("--no-wake", action="store_true", help="关闭唤醒词功能")
        parser.add_argument("--no-sv", action="store_true", help="关闭声纹识别功能")
        parser.add_argument("--add-wake", type=str, help="添加自定义唤醒词")
        parser.add_argument(
            "--register", action="store_true", help="强制进入声纹注册模式"
        )
        parser.add_argument("--register-name", type=str, help="注册声纹的用户名")
        parser.add_argument(
            "--max-temp", type=int, help="设置最大临时声纹数量", default=10
        )
        parser.add_argument("--debug", action="store_true", help="开启调试模式")
        args = parser.parse_args()

        if args.debug:
            os.environ["MODALITY_DEBUG"] = "1"
        else:
            os.environ["MODALITY_DEBUG"] = "0"

        print("正在初始化智能座舱语音识别系统...")
        self.speecher = SpeechRecognition(name="speech_recognition")
        print("语音识别模态创建成功")

        result = self.manager.register_modality(self.speecher)
        if result != SUCCESS:
            return

        print("语音识别模态注册成功")

        result = self.manager.start_modality("speech_recognition")
        if result != SUCCESS:
            return

        print("语音识别模态启动成功")

        if args.no_wake:
            self.speecher.toggle_wake_word(False)

        if args.add_wake:
            self.speecher.add_wake_word(args.add_wake)

        self.speecher.set_max_temp_speakers(args.max_temp)

        if args.register:
            name = args.register_name if args.register_name else "新用户"
            self.speecher.register_speaker(name)

        print("语音模态初始化完成")
        speecher_player.speech_synthesize_sync("语音模态初始化完成")
        logger.log("语音模态初始化完成")

    def init_headpose(self) -> None:
        "初始化视觉模态(头部姿态部分)"
        parser = argparse.ArgumentParser(description="驾驶员监测系统演示")
        parser.add_argument("--camera", type=int,
                            default=0, help="摄像头ID (默认为0)")
        parser.add_argument(
            "--width", type=int, default=640, help="图像宽度 (默认为640)"
        )
        parser.add_argument(
            "--height", type=int, default=480, help="图像高度 (默认为480)"
        )
        parser.add_argument(
            "--video", type=str, default="", help="使用视频文件而不是摄像头"
        )
        parser.add_argument("--record", type=str, default="", help="录制结果到视频文件")
        parser.add_argument(
            "--debug", action="store_true", help="开启调试模式，显示更详细信息"
        )
        parser.add_argument(
            "--method",
            type=str,
            default="gru",
            choices=["geom", "gru"],
            help="选择方法 (默认为gru)",
        )
        args = parser.parse_args()

        if args.debug:
            os.environ["MODALITY_DEBUG"] = "1"
        else:
            os.environ["MODALITY_DEBUG"] = "0"

        video_source = args.camera

        print("使用 GRU 模型进行头部姿态检测")

        self.headposer = HeadPoseTrackerGRU(
            source=video_source, width=args.width, height=args.height, debug=args.debug
        )

        result = self.manager.register_modality(self.headposer)
        if result != SUCCESS:
            return

        result = self.manager.start_modality(self.headposer.name)
        if result != SUCCESS:
            return

        print("头部姿态识别系统初始化完成")
        speecher_player.speech_synthesize_sync("视觉模态(头部姿态部分)初始化完成")
        logger.log("视觉模态(头部姿态部分)初始化完成")

    def init_static_gesture(self) -> None:
        "初始化视觉模态(手势动作部分)"
        # 解析命令行参数
        parser = argparse.ArgumentParser(description="手势识别模态演示")

        # 摄像头相关参数
        parser.add_argument(
            "--camera", type=int, default=0, help="摄像头设备ID (默认: 0)"
        )
        parser.add_argument(
            "--width", type=int, default=640, help="摄像头图像宽度 (默认: 640)"
        )
        parser.add_argument(
            "--height", type=int, default=480, help="摄像头图像高度 (默认: 480)"
        )

        # 模型相关参数
        path = "Modality/models/static_gesture_recognition/"
        parser.add_argument(
            "--model",
            type=str,
            default=path + "model_output/gesture_model.h5",
            help="手势识别模型路径",
        )
        parser.add_argument(
            "--feature-mean",
            type=str,
            default=path + "model_data/feature_mean.npy",
            help="特征均值文件路径",
        )
        parser.add_argument(
            "--feature-scale",
            type=str,
            default=path + "model_data/feature_scale.npy",
            help="特征缩放文件路径",
        )

        # 识别相关参数
        parser.add_argument(
            "--confidence",
            type=float,
            default=0.75,
            help="手势识别置信度阈值 (默认: 0.75)",
        )
        parser.add_argument(
            "--stability", type=float, default=0.7, help="手势稳定性阈值 (默认: 0.7)"
        )

        # 调试相关参数
        parser.add_argument("--debug", action="store_true", help="开启调试模式")
        args = parser.parse_args()

        # 如果模型目录不存在，创建它
        model_dir = os.path.dirname(args.model)
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)

        # 检查模型文件是否存在
        if (
            not os.path.exists(args.model)
            or not os.path.exists(args.feature_mean)
            or not os.path.exists(args.feature_scale)
        ):
            print("错误: 模型文件不存在。请确保以下文件存在:")
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
            os.environ["MODALITY_DEBUG"] = "1"

        print("初始化手势识别模态...")
        print(f"  - 摄像头: {args.camera}")
        print(f"  - 分辨率: {args.width}x{args.height}")
        print(f"  - 模型: {args.model}")
        print(f"  - 置信度阈值: {args.confidence}")
        print(f"  - 稳定性阈值: {args.stability}")

        # 创建模态管理器
        # manager = ModalityManager()

        # 创建手势识别模态
        self.static_gesture_tracker = GestureTracker(
            name="static_gesture_tracker",
            source=args.camera,
            width=args.width,
            height=args.height,
            model_path=args.model,
            feature_mean_path=args.feature_mean,
            feature_scale_path=args.feature_scale,
            confidence_threshold=args.confidence,
            stability_threshold=args.stability,
            debug=args.debug,
        )

        # 注册模态
        self.manager.register_modality(self.static_gesture_tracker)

        # 启动模态
        result = self.manager.start_modality("static_gesture_tracker")

        if result != 0:
            print(f"错误: 启动手势识别模态失败，错误码: {result}")
            return

        print("手势识别模态启动成功")
        logger.log("视觉模态(手势动作部分)初始化完成")
        speecher_player.speech_synthesize_sync("视觉模态(手势动作部分)初始化完成")

    def init_gazer(self) -> None:
        "初始化视觉模态(视线方向部分)"
        parser = argparse.ArgumentParser(description="视线方向跟踪演示")
        parser.add_argument(
            "--camera", type=int, default=0, help="摄像头设备ID (默认: 0)"
        )
        parser.add_argument(
            "--width", type=int, default=640, help="摄像头图像宽度 (默认: 640)"
        )
        parser.add_argument(
            "--height", type=int, default=480, help="摄像头图像高度 (默认: 480)"
        )
        parser.add_argument(
            "--min-detection-confidence",
            type=float,
            default=0.5,
            help="检测置信度阈值 (默认: 0.5)",
        )
        parser.add_argument(
            "--min-tracking-confidence",
            type=float,
            default=0.5,
            help="跟踪置信度阈值 (默认: 0.5)",
        )
        parser.add_argument(
            "--min-consensus",
            type=int,
            default=6,
            help="达成共识所需的最小样本数 (默认: 6)",
        )
        parser.add_argument(
            "--video", type=str, default="", help="使用视频文件而不是摄像头"
        )
        parser.add_argument("--record", type=str, default="", help="录制结果到视频文件")
        parser.add_argument("--debug", action="store_true", help="开启调试模式")

        args = parser.parse_args()
        if args.debug:
            os.environ["MODALITY_DEBUG"] = "1"
        else:
            os.environ["MODALITY_DEBUG"] = "0"

        source = args.video if args.video else args.camera

        self.gazer = GazeDirectionTracker(
            name="gaze_direction_tracker",
            source=source,
            width=args.width,
            height=args.height,
            min_detection_confidence=args.min_detection_confidence,
            min_tracking_confidence=args.min_tracking_confidence,
            debug=args.debug,
        )

        result = self.manager.register_modality(self.gazer)
        if result != SUCCESS:
            print(f"错误: 注册视线方向跟踪器失败，错误码: {result}")
            return

        result = self.manager.start_modality("gaze_direction_tracker")
        if result != SUCCESS:
            print(f"错误: 启动视线方向跟踪器失败，错误码: {result}")
            return

        logger.log("视觉模态(视线方向部分)初始化完成")
        speecher_player.speech_synthesize_sync("视觉模态(视线方向部分)初始化完成")

    def control(self) -> None:
        "协调各模态的工作"
        try:
            individuation = get_component("individuation")
            while True:
                key_info = self.manager.get_all_key_info()
                # print(key_info)
                for name, key_info in key_info.items():
                    # print(f"模态: {name}")
                    if name == "speech_recognition":
                        if key_info is None:
                            continue
                        print(f"语音识别结果: {key_info}")
                        individuation.speech_individuation(key_info)
                        time.sleep(1)
                    elif name == "head_pose_tracker_gru":
                        pass
                        # print(f"头部识别结果: {key_info}")
                    elif name == "static_gesture_tracker":
                        print(f"手势识别结果: {key_info}")
                        individuation.gesture_individuation(key_info)
                        time.sleep(1)
                    elif name == "gaze_direction_tracker":
                        pass
                        # print(f"视线方向识别结果: {key_info}")
                    else:
                        assert False, f"未知模态: {name}"

                    # time.sleep(1)
            # self.work_flag = True
        except KeyboardInterrupt:
            print("\n检测到终止信号")
        finally:
            print("正在关闭系统...")
            self.manager.shutdown_all()
            print("系统已关闭")
