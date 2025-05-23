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
    parser.add_argument("--show-gaze", action="store_true", help="显示视线方向信息（需GRU模型）")
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

def draw_gaze_direction(image, gaze_data):
    """绘制视线方向指示器"""
    if not gaze_data:
        return image
    
    h, w, _ = image.shape
    direction = gaze_data["direction"]
    confidence = gaze_data["confidence"]
    h_ratio = gaze_data["horizontal_ratio"]
    v_ratio = gaze_data["vertical_ratio"]
    
    left_eye_landmarks = gaze_data["left_eye"]["eye_landmarks"]
    right_eye_landmarks = gaze_data["right_eye"]["eye_landmarks"]
    
    for point in left_eye_landmarks:
        cv2.circle(image, (int(point[0]), int(point[1])), 1, (0, 255, 0), -1)
    for point in right_eye_landmarks:
        cv2.circle(image, (int(point[0]), int(point[1])), 1, (0, 255, 0), -1)
    
    center_x, center_y = w - 120, 130
    radius = 30
    
    cv2.circle(image, (center_x, center_y), radius, (30, 30, 30), -1)
    cv2.circle(image, (center_x, center_y), radius, (255, 255, 255), 1)
    
    line_length = int(radius * 0.8)
    end_x = center_x + int(h_ratio * line_length)
    end_y = center_y + int(v_ratio * line_length)

    cv2.line(image, (center_x, center_y), (end_x, end_y), (0, 255, 255), 2)
    cv2.circle(image, (end_x, end_y), 3, (0, 0, 255), -1)
    
    direction_text = f"Gaze: {direction}"
    cv2.putText(image, direction_text, (w - 240, 170), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    
    ratio_text = f"H: {h_ratio:.2f}, V: {v_ratio:.2f}"
    cv2.putText(image, ratio_text, (w - 240, 190), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    return image

def main():
    args = parse_args()
    
    if args.debug:
        os.environ["MODALITY_DEBUG"] = "1"
        logger.info("调试模式已开启")
    else:
        os.environ["MODALITY_DEBUG"] = "0"
    
    video_source = args.video if args.video else args.camera
    
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
    
    try:
        while True:
            states = manager.update_all()
            
            if not states or monitor.name not in states:
                logger.warning("无法获取驾驶员状态，退出中...")
                break
            
            driver_state = states[monitor.name]
            
            if driver_state.frame is not None:
                frame = driver_state.frame.copy()
                display_frame = frame.copy()
                
                if args.debug and driver_state.detections["head_pose"]["detected"]:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    rgb_frame.flags.writeable = False
                    results = face_mesh.process(rgb_frame)
                    rgb_frame.flags.writeable = True
                    display_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

                    display_frame = draw_face_mesh(display_frame, results)
                
                if driver_state.detections["head_pose"]["detected"]:
                    head_pose = driver_state.detections["head_pose"]
                    head_movement = driver_state.detections["head_movement"]
                    
                    # 获取视线方向数据（如果使用GRU模型）
                    gaze_direction = driver_state.detections.get("gaze_direction", None)
                    
                    h, w = display_frame.shape[:2]
                    info_panel_height = 120
                    info_panel = np.ones((info_panel_height, w, 3), dtype=np.uint8) * 240
                    
                    # 显示头部姿态
                    pitch = head_pose["pitch"]
                    yaw = head_pose["yaw"]
                    roll = head_pose["roll"]
                    pose_text = f"Head Pose: P={pitch:.2f} Y={yaw:.2f} R={roll:.2f}"
                    cv2.putText(info_panel, pose_text, (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                    
                    # 显示头部动作状态
                    is_nodding = head_movement["is_nodding"]
                    is_shaking = head_movement["is_shaking"]
                    movement_status = head_movement["status"]
                    
                    movement_text = f'Movement: Nodding={is_nodding}, Shaking={is_shaking}, Status={movement_status}'
                    cv2.putText(info_panel, movement_text, (10, 60), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                    
                    # 显示头部动作置信度
                    nod_conf = head_movement["nod_confidence"]
                    shake_conf = head_movement["shake_confidence"]
                    conf_text = f'Confidence: Nod={nod_conf:.2f}, Shake={shake_conf:.2f}'
                    cv2.putText(info_panel, conf_text, (10, 90), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

                    # 在主画面上显示当前状态
                    status_color = (0, 255, 0)  # 绿色
                    if movement_status == "nodding":
                        status_color = (0, 165, 255)  # 橙色
                    elif movement_status == "shaking":
                        status_color = (0, 0, 255)    # 红色
                        
                    cv2.putText(display_frame, f"Status: {movement_status}", (w - 250, 40), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

                    display_frame = draw_gaze_direction(display_frame, gaze_direction)

                    combined_frame = np.vstack((display_frame, info_panel))
                    
                    VisualizationUtil.show_frame("Driver Monitoring", combined_frame)
                    
                    if video_writer:
                        video_writer.write(frame)
                else:
                    cv2.putText(display_frame, "No Driver Detected", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    VisualizationUtil.show_frame("Driver Monitoring", display_frame)
            
            key = VisualizationUtil.wait_key(1)
            if key == ord('q') or key == ord('Q'):
                break
    
    except KeyboardInterrupt:
        logger.info("接收到中断信号，退出中...")
    
    finally:
        if face_mesh:
            face_mesh.close()
        
        result = manager.shutdown_all()
        if result != SUCCESS:
            logger.warning(f"关闭模态时遇到问题: {get_error_message(result)}")
        
        if video_writer:
            video_writer.release()
        VisualizationUtil.destroy_windows()

if __name__ == "__main__":
    main()