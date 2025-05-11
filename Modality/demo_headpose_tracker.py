import cv2
import numpy as np
import time
import argparse
import sys
import os
from typing import Dict, Any
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='driver_monitor.log',
    filemode='w'
)
logger = logging.getLogger('DriverMonitor')

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

from Modality.core import ModalityManager
from Modality.core.error_codes import SUCCESS, get_error_message
from Modality.utils.visualization import VisualizationUtil

def parse_args():
    parser = argparse.ArgumentParser(description="驾驶员监测系统演示")
    parser.add_argument("--camera", type=int, default=0, help="摄像头ID (默认为0)")
    parser.add_argument("--width", type=int, default=640, help="图像宽度 (默认为640)")
    parser.add_argument("--height", type=int, default=480, help="图像高度 (默认为480)")
    parser.add_argument("--video", type=str, default="", help="使用视频文件而不是摄像头")
    parser.add_argument("--record", type=str, default="", help="录制结果到视频文件")
    parser.add_argument("--debug", action="store_true", help="开启调试模式，显示更详细信息")
    parser.add_argument("--method", type=str, default="gru", choices=["geom", "gru"], help="选择方法 (默认为gru)")
    return parser.parse_args()

def draw_face_mesh(image, results):
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            try:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                )
                    
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style()
                )
            except Exception as e:
                logger.error(f"绘制面部网格时出错: {str(e)}")
    
    return image

def main():
    args = parse_args()
    
    if args.debug:
        os.environ["MODALITY_DEBUG"] = "1"
        logger.info("调试模式已开启")
    else:
        os.environ["MODALITY_DEBUG"] = "0"
    
    #video_source = args.video if args.video else args.camera
    video_source = args.camera
    
    logger.info(f"使用视频源: {video_source}")
    manager = ModalityManager()

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
    
    result = manager.register_modality(monitor)
    if result != SUCCESS:
        logger.error(f"注册驾驶员监测器失败: {get_error_message(result)}")
        return
    
    result = manager.start_modality(monitor.name)
    if result != SUCCESS:
        logger.error(f"启动驾驶员监测器失败: {get_error_message(result)}")
        return
    
    logger.info("驾驶员监测器已启动")
    
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
    
    print("按 'q' 键退出")
    
    from individuation import Individuation
    while True:
        states = manager.update_all()
        #print(states)
        state = states['head_pose_tracker_gru']
        if state.detections['head_movement']['is_nodding']:
            print("点头")
            #Individuation.head_individuation("点头")
        else:
            print("未检测到点头")
        if state.detections['head_movement']['is_shaking']:
            print("摇头")
            #Individuation.head_individuation("摇头")
        else:
            print("未检测到摇头")

        time.sleep(0.1)

if __name__ == "__main__":
    main()
 