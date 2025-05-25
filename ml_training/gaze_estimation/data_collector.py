import cv2
import numpy as np
import mediapipe as mp
import time
import json
import os
import socket
import argparse
import math
import datetime
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

GAZE_DIRECTIONS = {
    0: "center",
    1: "left",
    2: "right",
    3: "up",
    4: "down",
    5: "up_left",
    6: "up_right",
    7: "down_left",
    8: "down_right"
}

GAZE_DIRECTIONS_EN = {
    "center": "Center",
    "left": "Left",
    "right": "Right",
    "up": "Up",
    "down": "Down",
    "up_left": "Up-Left",
    "up_right": "Up-Right",
    "down_left": "Down-Left",
    "down_right": "Down-Right"
}

DEFAULT_DEVICE_ID = 0
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")

def put_chinese_text(img, text, position, font_size=30, color=(0, 255, 0)):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    try:
        font_paths = [
            os.path.join(os.environ.get('SYSTEMROOT', 'C:\\Windows'), 'Fonts', 'simhei.ttf'),
            os.path.join(os.environ.get('SYSTEMROOT', 'C:\\Windows'), 'Fonts', 'msyh.ttc'),
            os.path.join(os.environ.get('SYSTEMROOT', 'C:\\Windows'), 'Fonts', 'simsun.ttc'),
            "/System/Library/Fonts/PingFang.ttc"
        ]
        
        font = None
        for path in font_paths:
            if os.path.exists(path):
                font = ImageFont.truetype(path, font_size)
                break
        
        if font is None:
            font = ImageFont.load_default()
    except Exception as e:
        print(f"加载字体失败: {e}")
        font = ImageFont.load_default()
    
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

class GazeParams:
    """视线方向检测的参数常量"""
    LEFT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    RIGHT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    LEFT_IRIS_INDICES = [468, 469, 470, 471, 472]
    RIGHT_IRIS_INDICES = [473, 474, 475, 476, 477]

    ESSENTIAL_LANDMARKS = LEFT_EYE_INDICES + RIGHT_EYE_INDICES + LEFT_IRIS_INDICES + RIGHT_IRIS_INDICES

class GazeDataCollector:
    def __init__(self, 
                 device_id=DEFAULT_DEVICE_ID, 
                 width=DEFAULT_WIDTH, 
                 height=DEFAULT_HEIGHT,
                 output_dir=DEFAULT_OUTPUT_DIR):
        
        self.device_id = device_id
        self.width = width
        self.height = height
        self.output_dir = output_dir
        
        os.makedirs(self.output_dir, exist_ok=True)
        self.device_identifier = f"{socket.gethostname()}_{device_id}"
        self.cap = None
        self.face_mesh = None
        
        self.current_gaze_direction = None
        
        self.collected_counts = {direction: 0 for direction in GAZE_DIRECTIONS.values()}
        
        for direction in GAZE_DIRECTIONS.values():
            direction_dir = os.path.join(self.output_dir, direction)
            os.makedirs(direction_dir, exist_ok=True)
            
    def initialize(self):
        try:
            self.cap = cv2.VideoCapture(self.device_id, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(self.device_id)
                if not self.cap.isOpened():
                    raise RuntimeError(f"无法打开摄像头 {self.device_id}")
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            
            self.face_mesh = mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            return True
        except Exception as e:
            print(f"初始化失败: {str(e)}")
            return False
        
    def shutdown(self):
        if self.cap:
            self.cap.release()
            
        if self.face_mesh:
            self.face_mesh.close()
            
        cv2.destroyAllWindows()
    
    def start_capture(self, gaze_direction_idx):
        if gaze_direction_idx not in GAZE_DIRECTIONS:
            raise ValueError(f"Invalid gaze direction index: {gaze_direction_idx}")
            
        self.current_gaze_direction = GAZE_DIRECTIONS[gaze_direction_idx]
        print(f"Current capture direction set to: {GAZE_DIRECTIONS_EN[self.current_gaze_direction]}")
    
    def _calculate_iris_position(self, eye_landmarks, iris_landmarks):
        """
        计算虹膜相对于眼睛的位置
        
        Args:
            eye_landmarks: 眼睛轮廓关键点
            iris_landmarks: 虹膜关键点
            
        Returns:
            Tuple[float, float]: 水平和垂直位置比例 (-1到1)
        """
        eye_x_coords = [p[0] for p in eye_landmarks]
        eye_y_coords = [p[1] for p in eye_landmarks]
        
        eye_left = min(eye_x_coords)
        eye_right = max(eye_x_coords)
        eye_top = min(eye_y_coords)
        eye_bottom = max(eye_y_coords)
        
        eye_width = max(eye_right - eye_left, 1e-5)
        eye_height = max(eye_bottom - eye_top, 1e-5)
        
        iris_center_x = sum([p[0] for p in iris_landmarks]) / len(iris_landmarks)
        iris_center_y = sum([p[1] for p in iris_landmarks]) / len(iris_landmarks)

        horizontal_ratio = 2 * (iris_center_x - eye_left) / eye_width - 1
        vertical_ratio = 2 * (iris_center_y - eye_top) / eye_height - 1
        
        return horizontal_ratio, vertical_ratio
    
    def _process_frame(self, frame):
        if self.face_mesh is None:
            return {
                "timestamp": time.time(), 
                "detected": False,
                "landmarks": {},
                "gaze_data": {
                    "left_eye": {"iris_position": (0, 0), "landmarks": []},
                    "right_eye": {"iris_position": (0, 0), "landmarks": []}
                }
            }, None
        
        h, w, _ = frame.shape
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)
        
        frame_data = {
            "timestamp": time.time(),
            "detected": False,
            "landmarks": {},
            "gaze_data": {
                "left_eye": {"iris_position": (0, 0), "landmarks": []},
                "right_eye": {"iris_position": (0, 0), "landmarks": []}
            }
        }
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            frame_data["detected"] = True

            left_eye_landmarks = []
            right_eye_landmarks = []
            left_iris_landmarks = []
            right_iris_landmarks = []

            for idx, landmark in enumerate(face_landmarks.landmark):
                x, y, z = landmark.x, landmark.y, landmark.z
                pixel_x, pixel_y = int(x * w), int(y * h)

                frame_data["landmarks"][idx] = {
                    "x": x, 
                    "y": y, 
                    "z": z, 
                    "pixel_x": pixel_x, 
                    "pixel_y": pixel_y
                }
                
                if idx in GazeParams.LEFT_EYE_INDICES:
                    left_eye_landmarks.append((pixel_x, pixel_y))

                if idx in GazeParams.RIGHT_EYE_INDICES:
                    right_eye_landmarks.append((pixel_x, pixel_y))

                if idx in GazeParams.LEFT_IRIS_INDICES:
                    left_iris_landmarks.append((pixel_x, pixel_y))

                if idx in GazeParams.RIGHT_IRIS_INDICES:
                    right_iris_landmarks.append((pixel_x, pixel_y))

            if (len(left_eye_landmarks) == len(GazeParams.LEFT_EYE_INDICES) and
                len(right_eye_landmarks) == len(GazeParams.RIGHT_EYE_INDICES) and
                len(left_iris_landmarks) == len(GazeParams.LEFT_IRIS_INDICES) and
                len(right_iris_landmarks) == len(GazeParams.RIGHT_IRIS_INDICES)):
                
                left_h_ratio, left_v_ratio = self._calculate_iris_position(left_eye_landmarks, left_iris_landmarks)
                right_h_ratio, right_v_ratio = self._calculate_iris_position(right_eye_landmarks, right_iris_landmarks)

                frame_data["gaze_data"]["left_eye"]["iris_position"] = (left_h_ratio, left_v_ratio)
                frame_data["gaze_data"]["right_eye"]["iris_position"] = (right_h_ratio, right_v_ratio)
                frame_data["gaze_data"]["left_eye"]["landmarks"] = left_eye_landmarks
                frame_data["gaze_data"]["right_eye"]["landmarks"] = right_eye_landmarks
                frame_data["gaze_data"]["horizontal_ratio_avg"] = (left_h_ratio + right_h_ratio) / 2
                frame_data["gaze_data"]["vertical_ratio_avg"] = (left_v_ratio + right_v_ratio) / 2
        
        return frame_data, results
    
    def _display_frame(self, frame, results, frame_data):
        h, w, _ = frame.shape

        if results and results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]

            mp_drawing.draw_landmarks(
                frame,
                face_landmarks,
                mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )

            mp_drawing.draw_landmarks(
                frame,
                face_landmarks,
                mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style()
            )

            if frame_data["detected"]:
                left_iris = frame_data["gaze_data"]["left_eye"]["iris_position"]
                right_iris = frame_data["gaze_data"]["right_eye"]["iris_position"]
                
                cv2.putText(
                    frame, 
                    f"Left eye: H={left_iris[0]:.2f}, V={left_iris[1]:.2f}", 
                    (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, (0, 255, 255), 1
                )
                cv2.putText(
                    frame, 
                    f"Right eye: H={right_iris[0]:.2f}, V={right_iris[1]:.2f}", 
                    (10, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, (0, 255, 255), 1
                )

                h_avg = (left_iris[0] + right_iris[0]) / 2
                v_avg = (left_iris[1] + right_iris[1]) / 2
                cv2.putText(
                    frame, 
                    f"Avg: H={h_avg:.2f}, V={v_avg:.2f}", 
                    (10, 110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, (0, 255, 0), 2
                )
        
        if self.current_gaze_direction is not None:
            cv2.putText(
                frame, 
                f"Current direction: {GAZE_DIRECTIONS_EN[self.current_gaze_direction]}", 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, (0, 0, 255), 2
            )
            
            direction_indicator_size = 120
            margin = 20
            indicator_x = w - direction_indicator_size - margin
            indicator_y = margin
            
            self._draw_direction_indicator(
                frame, 
                self.current_gaze_direction,
                indicator_x,
                indicator_y,
                direction_indicator_size
            )
        else:
            cv2.putText(
                frame, 
                "Select direction with number keys", 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, (0, 255, 0), 2
            )

        y_pos = 160
        cv2.putText(
            frame, 
            f"Samples collected:", 
            (10, y_pos), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, (255, 255, 255), 2
        )
        y_pos += 30
        
        for direction, count in self.collected_counts.items():
            cv2.putText(
                frame, 
                f"{GAZE_DIRECTIONS_EN[direction]}: {count}", 
                (20, y_pos), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, (200, 200, 200), 1
            )
            y_pos += 25
        
        guide_text = ", ".join([f"{i}: {GAZE_DIRECTIONS_EN[dir]}" for i, dir in GAZE_DIRECTIONS.items()])
        guide_text += ", Space: Capture, Q: Quit"

        text_width = cv2.getTextSize(guide_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
        if text_width > w - 20:
            guide_text1 = ", ".join([f"{i}: {GAZE_DIRECTIONS_EN[dir]}" for i, dir in list(GAZE_DIRECTIONS.items())[:5]])
            guide_text2 = ", ".join([f"{i}: {GAZE_DIRECTIONS_EN[dir]}" for i, dir in list(GAZE_DIRECTIONS.items())[5:]])
            guide_text2 += ", Space: Capture, Q: Quit"
            
            cv2.putText(frame, guide_text1, (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, guide_text2, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            cv2.putText(frame, guide_text, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow("Gaze Direction Data Collector", frame)
    
    def _draw_direction_indicator(self, frame, direction, x, y, size):
        """绘制方向指示器"""
        center_x = x + size // 2
        center_y = y + size // 2
        radius = size // 2 - 5
        
        cv2.circle(frame, (center_x, center_y), radius, (30, 30, 30), -1)
        cv2.circle(frame, (center_x, center_y), radius, (150, 150, 150), 2)

        cv2.circle(frame, (center_x, center_y), 5, 
                 (0, 255, 0) if direction == "center" else (100, 100, 100), -1)

        point_dist = int(radius * 0.7)

        direction_points = {
            "left": (center_x - point_dist, center_y),
            "right": (center_x + point_dist, center_y),
            "up": (center_x, center_y - point_dist),
            "down": (center_x, center_y + point_dist),
            "up_left": (center_x - int(point_dist * 0.7), center_y - int(point_dist * 0.7)),
            "up_right": (center_x + int(point_dist * 0.7), center_y - int(point_dist * 0.7)),
            "down_left": (center_x - int(point_dist * 0.7), center_y + int(point_dist * 0.7)),
            "down_right": (center_x + int(point_dist * 0.7), center_y + int(point_dist * 0.7))
        }
        
        for dir_name, point in direction_points.items():
            color = (0, 0, 255) if dir_name == direction else (100, 100, 100)
            cv2.circle(frame, point, 10, color, -1)

        label_x = x + 5
        label_y = y + size + 20
        cv2.putText(
            frame, 
            f"Look at: {GAZE_DIRECTIONS_EN[direction]}", 
            (label_x, label_y), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, (0, 255, 255), 2
        )
    
    def _save_data(self, frame_data):
        if not frame_data["detected"]:
            print("No face detected, cannot save data")
            return
            
        if self.current_gaze_direction is None:
            print("No gaze direction selected, please select a direction first")
            return

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        direction_dir = os.path.join(self.output_dir, self.current_gaze_direction)

        json_filename = f"{self.device_identifier}_{timestamp}.json"
        json_filepath = os.path.join(direction_dir, json_filename)
        
        data = {
            "device": self.device_identifier,
            "timestamp": timestamp,
            "gaze_direction": self.current_gaze_direction,
            "landmarks": {str(k): v for k, v in frame_data["landmarks"].items()},
            "gaze_data": {
                "left_eye_iris_position": frame_data["gaze_data"]["left_eye"]["iris_position"],
                "right_eye_iris_position": frame_data["gaze_data"]["right_eye"]["iris_position"],
                "horizontal_ratio_avg": frame_data["gaze_data"]["horizontal_ratio_avg"],
                "vertical_ratio_avg": frame_data["gaze_data"]["vertical_ratio_avg"]
            }
        }
        
        with open(json_filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        self.collected_counts[self.current_gaze_direction] += 1
        
        print(f"Data saved: {self.current_gaze_direction}, Samples: {self.collected_counts[self.current_gaze_direction]}")
    
    def run(self):
        if not self.cap:
            if not self.initialize():
                print("Initialization failed, exiting...")
                return
            
        try:
            time.sleep(1)
            
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read from camera")
                    break
                
                frame_data, results = self._process_frame(frame)
                self._display_frame(frame, results, frame_data)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    break
                elif key in [ord(str(i)) for i in range(len(GAZE_DIRECTIONS))]:
                    gaze_idx = int(chr(key))
                    self.start_capture(gaze_idx)
                elif key == 32:
                    if self.current_gaze_direction is not None:
                        self._save_data(frame_data)
                
        finally:
            self.shutdown()

def main():
    parser = argparse.ArgumentParser(description="Gaze Direction Data Collection Tool")
    parser.add_argument('-c', '--camera', type=int, default=DEFAULT_DEVICE_ID, 
                        help=f"Camera ID (default: {DEFAULT_DEVICE_ID})")
    parser.add_argument('-W', '--width', type=int, default=DEFAULT_WIDTH,
                        help=f"Image width (default: {DEFAULT_WIDTH})")
    parser.add_argument('-H', '--height', type=int, default=DEFAULT_HEIGHT,
                        help=f"Image height (default: {DEFAULT_HEIGHT})")
    parser.add_argument('-o', '--output', type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})")
    
    args = parser.parse_args()
    
    collector = GazeDataCollector(
        device_id=args.camera,
        width=args.width,
        height=args.height,
        output_dir=args.output
    )
    
    try:
        print("Gaze Direction Data Collection Tool")
        print("Please press a number key to select gaze direction:")
        for idx, direction in GAZE_DIRECTIONS.items():
            print(f"  {idx} - {GAZE_DIRECTIONS_EN[direction]}")
        print("When collecting data, face the camera, keep your head still, and only move your eyes")
        print("Press Space to capture the current image, press Q to exit")
        
        collector.run()
        
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        collector.shutdown()

if __name__ == "__main__":
    main()
