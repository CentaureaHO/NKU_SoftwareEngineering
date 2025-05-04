#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#_author = 'Yidian Lin'

"""
Module Description:
    用于实现车载多模态智能交互系统的控制功能
"""

import os
import threading
from typing import List
from enum import Enum
from applications import application
#from Modality.demo_speech_recognition import main as demo_main
import os
import sys
import time
import argparse
import logging
from Modality.core.modality_manager import ModalityManager
from Modality.speech.speech_recognition import SpeechRecognition
from Modality.core.error_codes import SUCCESS, get_error_message
import cv2
import numpy as np
import time
import argparse
import sys
import os
from typing import Dict, Any
import logging
import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

from Modality.core import ModalityManager
from Modality.core.error_codes import SUCCESS, get_error_message
from Modality.utils.visualization import VisualizationUtil


class MultimodalController:
    # TODO():这个函数负责启动时打开各个模态,目前已做了语音
    def __init__(self) -> None:
        # 启动语音模态
        self.manager = ModalityManager()
        self.init_speecher()
        self.init_headpose()

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
        from individuation import Individuation
        if state.detections['head_movement']['is_nodding']:
            print("点头")
            Individuation.head_individuation("点头")
            time.sleep(0.5)
        elif state.detections['head_movement']['is_shaking']:
            print("摇头")
            Individuation.head_individuation("摇头")
            time.sleep(0.5)

    # TODO():
    def control(self) -> None:
        try:
            
            
            while True:
                # state = self.speech_modality.update()
                states = self.manager.update_all()
                for name, state in states.items():
                    print(f"模态: {name}")
                    if name == "speech_recognition":
                        self.control_speecher(state)
                    elif name == "head_pose_tracker_gru":
                        self.control_headpose(state)
                    else:
                        assert False, f"未知模态: {name}"
                time.sleep(0.1)
                """
                state = state_all.get("speech_recognition")
                if state and state.recognition["text"]:
                    current_text = state.recognition["text"]
                    
                    if current_text != last_recognized_text:
                        last_recognized_text = current_text
                        
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
                
                import msvcrt
                if msvcrt.kbhit():
                    cmd = msvcrt.getch().decode('utf-8', errors='ignore').lower()
                    print(f"\n输入命令: {cmd}")
                    
                    if cmd == 'r':
                        name = input("请输入用户名: ")
                        self.speech_modality.register_speaker(name)
                    elif cmd == 'd':
                        speakers = self.speech_modality.get_registered_speakers()
                        if not speakers:
                            print("没有已注册的声纹")
                            continue
                            
                        print("已注册声纹:")
                        for idx, (speaker_id, info) in enumerate(speakers.items()):
                            print(f"  {idx+1}. {info['name']} (ID: {speaker_id})")
                            
                        choice = input("请输入要删除的声纹编号: ")
                        try:
                            idx = int(choice) - 1
                            if 0 <= idx < len(speakers):
                                speaker_id = list(speakers.keys())[idx]
                                if self.speech_modality.delete_speaker(speaker_id):
                                    print(f"已删除声纹 {speakers[speaker_id]['name']}")
                            else:
                                print("无效的编号")
                        except ValueError:
                            print("请输入有效的数字")
                    elif cmd == 'w':
                        enabled = self.speech_modality.toggle_wake_word()
                        print(f"唤醒词功能已{'启用' if enabled else '关闭'}")
                    elif cmd == 'a':
                        wake_word = input("请输入要添加的唤醒词: ")
                        if self.speech_modality.add_wake_word(wake_word):
                            print(f"已添加唤醒词: {wake_word}")
                    elif cmd == 'k':
                        keyword = input("请输入要添加的关键词: ")
                        category = input("请输入关键词类别: ")
                        if self.speech_modality.add_keyword(keyword, category):
                            print(f"已添加关键词: {keyword} -> {category}")
                    elif cmd == 't':
                        temp_speakers = self.speech_modality.get_temp_speakers()
                        if not temp_speakers:
                            print("没有临时声纹")
                        else:
                            print("临时声纹列表:")
                            for temp_id, info in temp_speakers.items():
                                print(f"  {info['name']} (ID: {temp_id}, 创建时间: {info.get('created', 'N/A')})")
                                
                            promote = input("是否提升某个临时声纹为正式声纹? (y/n): ").lower()
                            if promote == 'y':
                                temp_id = input("请输入临时声纹ID: ")
                                if temp_id in temp_speakers:
                                    name = input("请输入新的用户名 (直接回车使用原名): ")
                                    name = name if name else None
                                    new_id = self.speech_modality.promote_temp_speaker(temp_id, name)
                                    if new_id:
                                        print(f"临时声纹已提升为正式声纹，新ID: {new_id}")
                                else:
                                    print(f"临时声纹 {temp_id} 不存在")
                    elif cmd == 's':
                        speakers = self.speech_modality.get_registered_speakers()
                        if not speakers:
                            print("没有已注册的声纹")
                        else:
                            print("已注册声纹列表:")
                            for speaker_id, info in speakers.items():
                                print(f"  {info['name']} (ID: {speaker_id})")
                    elif cmd == 'q':
                        break
                """
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