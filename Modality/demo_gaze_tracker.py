import cv2
import numpy as np
import time
import argparse
import sys
import os
from typing import Dict, Any
import logging
from PIL import Image, ImageDraw, ImageFont

logging.basicConfig(
    level=logging.DEBUG if os.environ.get('MODALITY_DEBUG', '0') == '1' else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='gaze_direction_tracker_demo.log',
    filemode='w'
)
logger = logging.getLogger('GazeDirectionDemo')

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from Modality.visual.gaze_direction_tracker import (
    DIRECTION_CENTER, DIRECTION_LEFT, DIRECTION_RIGHT, DIRECTION_UP, DIRECTION_DOWN,
    DIRECTION_UP_LEFT, DIRECTION_UP_RIGHT, DIRECTION_DOWN_LEFT, DIRECTION_DOWN_RIGHT
)
from Modality.core import ModalityManager
from Modality.core.error_codes import SUCCESS, get_error_message

COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_BLUE = (255, 0, 0)
COLOR_YELLOW = (0, 255, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_PURPLE = (255, 0, 255)
COLOR_ORANGE = (0, 165, 255)

def put_chinese_text(img, text, position, font_size=30, color=(0, 255, 0)):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    try:
        font_paths = [
            os.path.join(os.environ.get('SYSTEMROOT', 'C:\\Windows'), 'Fonts', 'simhei.ttf'),
            os.path.join(os.environ.get('SYSTEMROOT', 'C:\\Windows'), 'Fonts', 'msyh.ttc'),
            os.path.join(os.environ.get('SYSTEMROOT', 'C:\\Windows'), 'Fonts', 'simsun.ttc'),
        ]
        
        font = None
        for path in font_paths:
            if os.path.exists(path):
                font = ImageFont.truetype(path, font_size)
                break
        
        if font is None:
            font = ImageFont.load_default()
    except Exception as e:
        logger.error(f"加载字体出错：{e}")
        font = ImageFont.load_default()
    
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def parse_arguments():
    parser = argparse.ArgumentParser(description='视线方向跟踪演示')
    parser.add_argument('--camera', type=int, default=0, help='摄像头设备ID (默认: 0)')
    parser.add_argument('--width', type=int, default=640, help='摄像头图像宽度 (默认: 640)')
    parser.add_argument('--height', type=int, default=480, help='摄像头图像高度 (默认: 480)')
    parser.add_argument('--min-detection-confidence', type=float, default=0.5, help='检测置信度阈值 (默认: 0.5)')
    parser.add_argument('--min-tracking-confidence', type=float, default=0.5, help='跟踪置信度阈值 (默认: 0.5)')
    parser.add_argument('--min-consensus', type=int, default=6, help='达成共识所需的最小样本数 (默认: 6)')
    parser.add_argument('--video', type=str, default='', help='使用视频文件而不是摄像头')
    parser.add_argument('--record', type=str, default='', help='录制结果到视频文件')
    parser.add_argument('--debug', action='store_true', help='开启调试模式')
    
    return parser.parse_args()

def draw_direction_indicator(frame, direction, confidence):
    h, w = frame.shape[:2]
    center_x, center_y = w // 2, h // 2
    radius = min(w, h) // 6
    
    cv2.circle(frame, (center_x, center_y), radius, (30, 30, 30), -1)
    cv2.circle(frame, (center_x, center_y), radius, COLOR_WHITE, 2)
    
    arrow_length = int(radius * 0.8)
    arrow_thickness = 2
    diagonal_length = int(arrow_length * 0.7)  # 对角线箭头长度为主箭头的70%
    
    # 中心点
    cv2.circle(frame, (center_x, center_y), 5, COLOR_GREEN if direction == DIRECTION_CENTER else COLOR_WHITE, -1)
    
    # 上方
    cv2.arrowedLine(
        frame, 
        (center_x, center_y), 
        (center_x, center_y - arrow_length), 
        COLOR_RED if direction == DIRECTION_UP else COLOR_WHITE, 
        arrow_thickness
    )
    
    # 下方
    cv2.arrowedLine(
        frame, 
        (center_x, center_y), 
        (center_x, center_y + arrow_length), 
        COLOR_RED if direction == DIRECTION_DOWN else COLOR_WHITE, 
        arrow_thickness
    )
    
    # 左方
    cv2.arrowedLine(
        frame, 
        (center_x, center_y), 
        (center_x - arrow_length, center_y), 
        COLOR_RED if direction == DIRECTION_LEFT else COLOR_WHITE, 
        arrow_thickness
    )
    
    # 右方
    cv2.arrowedLine(
        frame, 
        (center_x, center_y), 
        (center_x + arrow_length, center_y), 
        COLOR_RED if direction == DIRECTION_RIGHT else COLOR_WHITE, 
        arrow_thickness
    )
    
    # 左上
    cv2.arrowedLine(
        frame, 
        (center_x, center_y), 
        (center_x - diagonal_length, center_y - diagonal_length), 
        COLOR_PURPLE if direction == DIRECTION_UP_LEFT else COLOR_WHITE, 
        arrow_thickness
    )
    
    # 右上
    cv2.arrowedLine(
        frame, 
        (center_x, center_y), 
        (center_x + diagonal_length, center_y - diagonal_length), 
        COLOR_PURPLE if direction == DIRECTION_UP_RIGHT else COLOR_WHITE, 
        arrow_thickness
    )
    
    # 左下
    cv2.arrowedLine(
        frame, 
        (center_x, center_y), 
        (center_x - diagonal_length, center_y + diagonal_length), 
        COLOR_PURPLE if direction == DIRECTION_DOWN_LEFT else COLOR_WHITE, 
        arrow_thickness
    )
    
    # 右下
    cv2.arrowedLine(
        frame, 
        (center_x, center_y), 
        (center_x + diagonal_length, center_y + diagonal_length), 
        COLOR_PURPLE if direction == DIRECTION_DOWN_RIGHT else COLOR_WHITE, 
        arrow_thickness
    )
    
    bar_width = int(radius * 2 * confidence)
    bar_height = 10
    bar_x = center_x - radius
    bar_y = center_y + radius + 20
    
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + radius * 2, bar_y + bar_height), COLOR_WHITE, 1)
    if bar_width > 0:
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), COLOR_GREEN, -1)

    conf_text = f"置信度: {confidence:.2f}"
    cv2.putText(
        frame, conf_text, 
        (bar_x, bar_y + bar_height + 20), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.6, COLOR_WHITE, 1
    )
    
    return frame

def draw_eye_gaze(frame, gaze_data):
    left_eye_landmarks = gaze_data["left_eye"]["eye_landmarks"]
    if left_eye_landmarks:
        for point in left_eye_landmarks:
            cv2.circle(frame, (int(point[0]), int(point[1])), 1, COLOR_GREEN, -1)
        
        left_iris_position = gaze_data["left_eye"]["iris_position"]
        h_ratio, v_ratio = left_iris_position
        left_eye_min_x = min([p[0] for p in left_eye_landmarks])
        left_eye_max_x = max([p[0] for p in left_eye_landmarks])
        left_eye_min_y = min([p[1] for p in left_eye_landmarks])
        left_eye_max_y = max([p[1] for p in left_eye_landmarks])
        
        left_eye_width = left_eye_max_x - left_eye_min_x
        left_eye_height = left_eye_max_y - left_eye_min_y
        
        left_iris_x = int(left_eye_min_x + (left_eye_width * (h_ratio + 1) / 2))
        left_iris_y = int(left_eye_min_y + (left_eye_height * (v_ratio + 1) / 2))
        
        cv2.circle(frame, (left_iris_x, left_iris_y), 3, COLOR_RED, -1)
    
    right_eye_landmarks = gaze_data["right_eye"]["eye_landmarks"]
    if right_eye_landmarks:
        for point in right_eye_landmarks:
            cv2.circle(frame, (int(point[0]), int(point[1])), 1, COLOR_GREEN, -1)
        
        right_iris_position = gaze_data["right_eye"]["iris_position"]
        h_ratio, v_ratio = right_iris_position
        right_eye_min_x = min([p[0] for p in right_eye_landmarks])
        right_eye_max_x = max([p[0] for p in right_eye_landmarks])
        right_eye_min_y = min([p[1] for p in right_eye_landmarks])
        right_eye_max_y = max([p[1] for p in right_eye_landmarks])
        
        right_eye_width = right_eye_max_x - right_eye_min_x
        right_eye_height = right_eye_max_y - right_eye_min_y
        
        right_iris_x = int(right_eye_min_x + (right_eye_width * (h_ratio + 1) / 2))
        right_iris_y = int(right_eye_min_y + (right_eye_height * (v_ratio + 1) / 2))
        
        cv2.circle(frame, (right_iris_x, right_iris_y), 3, COLOR_RED, -1)
    
    return frame

def main():
    """主函数"""
    args = parse_arguments()
    
    if args.debug:
        os.environ['MODALITY_DEBUG'] = '1'
    else:
        os.environ['MODALITY_DEBUG'] = '0'
    
    print("初始化视线方向跟踪器...")
    print(f"  - 视频源: {'摄像头 ' + str(args.camera) if not args.video else '视频文件 ' + args.video}")
    print(f"  - 分辨率: {args.width}x{args.height}")
    print(f"  - 共识阈值: {args.min_consensus}")
    
    manager = ModalityManager()
    
    source = args.video if args.video else args.camera
    
    from Modality.visual import GazeDirectionTracker
    
    gaze_tracker = GazeDirectionTracker(
        name="gaze_direction_tracker",
        source=source,
        width=args.width,
        height=args.height,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
        debug=args.debug
    )
    
    result = manager.register_modality(gaze_tracker)
    if result != SUCCESS:
        print(f"错误: 注册视线方向跟踪器失败，错误码: {result}")
        return
    
    result = manager.start_modality("gaze_direction_tracker")
    if result != SUCCESS:
        print(f"错误: 启动视线方向跟踪器失败，错误码: {result}")
        return
    
    print("视线方向跟踪器启动成功")
    print("按ESC键退出")
    
    video_writer = None
    if args.record:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            args.record,
            fourcc,
            30,
            (args.width, args.height)
        )
    
    # 更新方向名称字典以包含所有方向
    direction_names = {
        DIRECTION_CENTER: "中央",
        DIRECTION_LEFT: "左侧",
        DIRECTION_RIGHT: "右侧",
        DIRECTION_UP: "上方",
        DIRECTION_DOWN: "下方",
        DIRECTION_UP_LEFT: "左上",
        DIRECTION_UP_RIGHT: "右上",
        DIRECTION_DOWN_LEFT: "左下",
        DIRECTION_DOWN_RIGHT: "右下"
    }
    
    try:
        last_time = time.time()
        while True:
            current_time = time.time()
            fps = 1 / (current_time - last_time) if (current_time - last_time) > 0 else 0
            last_time = current_time

            states = manager.update_all()

            if "gaze_direction_tracker" in states:
                gaze_state = states["gaze_direction_tracker"]
                frame = gaze_state.frame
                
                if frame is not None:
                    display_frame = frame.copy()
                    
                    gaze_info = gaze_state.detections["gaze_direction"]
 
                    display_frame = put_chinese_text(
                        display_frame, 
                        f"FPS: {int(fps)}", 
                        (10, 30), 
                        font_size=24, 
                        color=COLOR_GREEN
                    )
                    
                    if gaze_info["face_detected"]:
                        direction = gaze_info["direction"]
                        confidence = gaze_info["confidence"]
                        h_ratio = gaze_info["horizontal_ratio"]
                        v_ratio = gaze_info["vertical_ratio"]

                        display_frame = draw_eye_gaze(display_frame, gaze_info)

                        display_frame = draw_direction_indicator(display_frame, direction, confidence)

                        direction_name = direction_names.get(direction, "未知")
                        text = f"视线方向: {direction_name}"
                        display_frame = put_chinese_text(
                            display_frame, 
                            text, 
                            (10, 70), 
                            font_size=28, 
                            color=COLOR_YELLOW
                        )

                        ratio_text = f"水平: {h_ratio:.2f}, 垂直: {v_ratio:.2f}"
                        display_frame = put_chinese_text(
                            display_frame, 
                            ratio_text, 
                            (10, 110), 
                            font_size=24, 
                            color=COLOR_WHITE
                        )
                    else:
                        display_frame = put_chinese_text(
                            display_frame, 
                            "未检测到人脸", 
                            (10, 70), 
                            font_size=28, 
                            color=COLOR_RED
                        )
                    
                    guide = "按ESC键退出"
                    display_frame = put_chinese_text(
                        display_frame, 
                        guide, 
                        (10, display_frame.shape[0] - 30), 
                        font_size=22, 
                        color=COLOR_WHITE
                    )
                    
                    if video_writer is not None:
                        video_writer.write(display_frame)
                    
                    cv2.imshow("视线方向跟踪演示", display_frame)

                key = cv2.waitKey(1)
                if key == 27:  # ESC
                    break
            else:
                print("未收到视线方向状态更新")
                time.sleep(0.01)
    
    except KeyboardInterrupt:
        print("程序被用户中断")
    except Exception as e:
        print(f"发生错误: {str(e)}")
    finally:
        manager.shutdown_all()
        
        if video_writer is not None:
            video_writer.release()
            
        cv2.destroyAllWindows()
        print("程序已退出")

if __name__ == "__main__":
    main()
