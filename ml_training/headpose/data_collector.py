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

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

class HeadPoseParams:
    LANDMARK_NOSE = 1
    LANDMARK_CHIN = 152
    LANDMARK_LEFT_EYE = 159
    LANDMARK_RIGHT_EYE = 386
    LANDMARK_LEFT_EAR = 234
    LANDMARK_RIGHT_EAR = 454
    LANDMARK_LEFT_FACE = 206
    LANDMARK_RIGHT_FACE = 426
    
    ESSENTIAL_LANDMARKS = [LANDMARK_NOSE, 33, 133, LANDMARK_LEFT_EYE, 145, 
                          263, 362, 374, LANDMARK_RIGHT_EYE, 473, 468]

GESTURE_TYPES = {
    0: "stationary",
    1: "nodding",
    2: "shaking",
    3: "other"
}

DEFAULT_DEVICE_ID = 0
DEFAULT_DURATION = 3
DEFAULT_FPS = 30
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")

class HeadPoseDataCollector:
    def __init__(self, 
                 device_id=DEFAULT_DEVICE_ID, 
                 width=DEFAULT_WIDTH, 
                 height=DEFAULT_HEIGHT,
                 fps=DEFAULT_FPS,
                 output_dir=DEFAULT_OUTPUT_DIR):
        
        self.device_id = device_id
        self.width = width
        self.height = height
        self.fps = fps
        self.output_dir = output_dir
        
        os.makedirs(self.output_dir, exist_ok=True)
        self.device_identifier = f"{socket.gethostname()}_{device_id}"
        self.cap = None
        self.face_mesh = None

        self.face_3d_coords = np.array([
            [285, 528, 200],
            [285, 371, 152],
            [197, 574, 128],
            [173, 425, 108],
            [360, 574, 128],
            [391, 425, 108]
        ], dtype=np.float64)
        
        self.frames_data = []
        self.is_recording = False
        self.start_time = 0
        self.frame_count = 0
        self.gesture_type = "stationary"
        
        self.countdown = 0
        self.countdown_start = 0
        
    def initialize(self):
        try:
            self.cap = cv2.VideoCapture(self.device_id, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(self.device_id)
                if not self.cap.isOpened():
                    raise RuntimeError(f"无法打开摄像头 {self.device_id}")
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
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
    
    def start_recording(self, gesture_type_idx, duration=DEFAULT_DURATION):
        if gesture_type_idx not in GESTURE_TYPES:
            raise ValueError(f"无效的姿态类型索引: {gesture_type_idx}")
            
        self.gesture_type = GESTURE_TYPES[gesture_type_idx]
        self.frames_data = []
        self.countdown = 1
        self.countdown_start = time.time()
        self.duration = duration
        print(f"准备录制 {self.gesture_type} 动作，{self.countdown}秒后开始...")
    
    def _rotation_matrix_to_angles(self, rotation_matrix):
        
        x = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        y = math.atan2(-rotation_matrix[2, 0], math.sqrt(rotation_matrix[0, 0] ** 2 +
                                                     rotation_matrix[1, 0] ** 2))
        z = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        return np.array([x, y, z]) * 180. / math.pi
    
    def _euclidean_dist(self, point1, point2):
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
        
    def _process_frame(self, frame):
        if self.face_mesh is None:
            return {"timestamp": time.time(), "detected": False, "landmarks": [], 
                    "key_data": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0, 
                                "nose_chin_distance": 0.0, "left_cheek_width": 0.0, 
                                "right_cheek_width": 0.0, "landmark_positions": []},
                    "raw_data": []}, None
        
        h, w, _ = frame.shape
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)
        
        frame_data = {
            "timestamp": time.time(),
            "detected": False,
            "landmarks": [],
            "key_data": {
                "pitch": 0.0,
                "yaw": 0.0,
                "roll": 0.0,
                "nose_chin_distance": 0.0,
                "left_cheek_width": 0.0,
                "right_cheek_width": 0.0,
                "landmark_positions": []
            },
            "raw_data": []
        }
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            frame_data["detected"] = True
            
            face_coordination_in_image = []
            landmarks_3d = {}
            raw_landmarks = []
            for idx, landmark in enumerate(face_landmarks.landmark):
                x, y, z = landmark.x, landmark.y, landmark.z
                x_px, y_px = int(x * w), int(y * h)
    
                raw_landmarks.append({
                    "index": idx,
                    "x": x,
                    "y": y, 
                    "z": z,
                    "x_px": x_px,
                    "y_px": y_px
                })
            frame_data["raw_data"] = raw_landmarks

            for idx, landmark in enumerate(face_landmarks.landmark):
                x, y, z = landmark.x, landmark.y, landmark.z
                x_px, y_px = int(x * w), int(y * h)
                
                landmarks_3d[idx] = (x, y, z)
                
                if idx in [1, 9, 57, 130, 287, 359]:
                    face_coordination_in_image.append([x_px, y_px])
                
            landmarks_list = []
            for idx in HeadPoseParams.ESSENTIAL_LANDMARKS:
                if idx in landmarks_3d:
                    landmarks_list.append(landmarks_3d[idx])
            
            frame_data["landmarks"] = landmarks_list
            frame_data["key_data"]["landmark_positions"] = [
                (landmarks_3d.get(HeadPoseParams.LANDMARK_NOSE, (0, 0, 0))),
                (landmarks_3d.get(HeadPoseParams.LANDMARK_CHIN, (0, 0, 0))),
                (landmarks_3d.get(HeadPoseParams.LANDMARK_LEFT_EYE, (0, 0, 0))),
                (landmarks_3d.get(HeadPoseParams.LANDMARK_RIGHT_EYE, (0, 0, 0))),
                (landmarks_3d.get(HeadPoseParams.LANDMARK_LEFT_EAR, (0, 0, 0))),
                (landmarks_3d.get(HeadPoseParams.LANDMARK_RIGHT_EAR, (0, 0, 0))),
                (landmarks_3d.get(HeadPoseParams.LANDMARK_LEFT_FACE, (0, 0, 0))),
                (landmarks_3d.get(HeadPoseParams.LANDMARK_RIGHT_FACE, (0, 0, 0)))
            ]
            
            if len(face_coordination_in_image) == 6:
                face_coordination_in_image = np.array(face_coordination_in_image, dtype=np.float64)
                
                focal_length = 1 * w
                cam_matrix = np.array([
                    [focal_length, 0, w / 2],
                    [0, focal_length, h / 2],
                    [0, 0, 1]
                ])
                dist_matrix = np.zeros((4, 1), dtype=np.float64)
                
                try:
                    success, rotation_vec, translation_vec = cv2.solvePnP(
                        self.face_3d_coords, 
                        face_coordination_in_image,
                        cam_matrix, 
                        dist_matrix
                    )
                    
                    if success:
                        rotation_matrix, _ = cv2.Rodrigues(rotation_vec)
                        angles = self._rotation_matrix_to_angles(rotation_matrix)
                        
                        frame_data["key_data"]["pitch"] = float(angles[0])
                        frame_data["key_data"]["yaw"] = float(angles[1])
                        frame_data["key_data"]["roll"] = float(angles[2])
                        
                        nose_point = None
                        chin_point = None
                        left_ear_point = None
                        left_face_point = None
                        right_ear_point = None
                        right_face_point = None
                        
                        for idx, lm in enumerate(face_landmarks.landmark):
                            x_px, y_px = int(lm.x * w), int(lm.y * h)
                            
                            if idx == HeadPoseParams.LANDMARK_NOSE:
                                nose_point = (x_px, y_px)
                            elif idx == HeadPoseParams.LANDMARK_CHIN:
                                chin_point = (x_px, y_px)
                            elif idx == HeadPoseParams.LANDMARK_LEFT_EAR:
                                left_ear_point = (x_px, y_px)
                            elif idx == HeadPoseParams.LANDMARK_LEFT_FACE:
                                left_face_point = (x_px, y_px)
                            elif idx == HeadPoseParams.LANDMARK_RIGHT_EAR:
                                right_ear_point = (x_px, y_px)
                            elif idx == HeadPoseParams.LANDMARK_RIGHT_FACE:
                                right_face_point = (x_px, y_px)
                        
                        if nose_point and chin_point:
                            frame_data["key_data"]["nose_chin_distance"] = self._euclidean_dist(nose_point, chin_point)
                            
                        if left_ear_point and left_face_point:
                            frame_data["key_data"]["left_cheek_width"] = self._euclidean_dist(left_ear_point, left_face_point)
                            
                        if right_ear_point and right_face_point:
                            frame_data["key_data"]["right_cheek_width"] = self._euclidean_dist(right_ear_point, right_face_point)
                
                except Exception as e:
                    print(f"计算头部姿态时出错: {str(e)}")
        
        return frame_data, results
    
    def _display_frame(self, frame, results, frame_data):
        h, w, _ = frame.shape
        
        if results and results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
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
                    mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                )
        
        pitch = frame_data["key_data"]["pitch"]
        yaw = frame_data["key_data"]["yaw"]
        roll = frame_data["key_data"]["roll"]
        
        if self.is_recording:
            remaining = self.start_time + self.duration - time.time()
            if remaining > 0:
                cv2.putText(frame, f"Recording {self.gesture_type}: {remaining:.1f}s", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.circle(frame, (20, 20), 10, (0, 0, 255), -1)
            else:
                self.is_recording = False
                self._save_data()
        elif self.countdown > 0:
            elapsed = time.time() - self.countdown_start
            remaining = self.countdown - elapsed
            if remaining > 0:
                cv2.putText(frame, f"Starting in {remaining:.1f}s", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            else:
                self.countdown = 0
                self.is_recording = True
                self.start_time = time.time()
                self.frame_count = 0
                print(f"开始录制 {self.gesture_type} 动作，持续 {self.duration} 秒...")
        else:
            cv2.putText(frame, "Ready (press 0-3 to record)", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(frame, f"Pitch: {pitch:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"Yaw: {yaw:.1f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"Roll: {roll:.1f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        cv2.putText(frame, "0: Stationary, 1: Nodding, 2: Shaking, 3: Other, Q: Quit", 
                   (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow("Head Pose Data Collector", frame)
    
    def _save_data(self):
        if not self.frames_data:
            print("没有数据可以保存")
            return
            
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.device_identifier}_{timestamp}_{self.gesture_type}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        data = {
            "device": self.device_identifier,
            "timestamp": timestamp,
            "gesture_type": self.gesture_type,
            "frame_count": len(self.frames_data),
            "fps": self.fps,
            "frames": self.frames_data
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        print(f"数据已保存到 {filepath}")
        print(f"共记录 {len(self.frames_data)} 帧 ({len(self.frames_data) / self.fps:.1f} 秒)")
        
    def run(self):
        if not self.cap:
            if not self.initialize():
                print("初始化失败，退出程序")
                return
            
        try:
            time.sleep(1)
            
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("无法从摄像头读取")
                    break
                
                frame_data, results = self._process_frame(frame)
                if self.is_recording:
                    self.frames_data.append(frame_data)
                    
                self._display_frame(frame, results, frame_data)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key in [ord('0'), ord('1'), ord('2'), ord('3')]:
                    gesture_idx = int(chr(key))
                    self.start_recording(gesture_idx, self.duration)
                
        finally:
            self.shutdown()
                

def main():
    parser = argparse.ArgumentParser(description="头部姿态数据收集")
    parser.add_argument('-c', '--camera', type=int, default=DEFAULT_DEVICE_ID, 
                        help=f"摄像头ID (默认: {DEFAULT_DEVICE_ID})")
    parser.add_argument('-d', '--duration', type=int, default=DEFAULT_DURATION,
                        help=f"每次录制的持续时间（秒）(默认: {DEFAULT_DURATION})")
    parser.add_argument('-f', '--fps', type=int, default=DEFAULT_FPS,
                        help=f"帧率 (默认: {DEFAULT_FPS})")
    parser.add_argument('-W', '--width', type=int, default=DEFAULT_WIDTH,
                        help=f"画面宽度 (默认: {DEFAULT_WIDTH})")
    parser.add_argument('-H', '--height', type=int, default=DEFAULT_HEIGHT,
                        help=f"画面高度 (默认: {DEFAULT_HEIGHT})")
    parser.add_argument('-o', '--output', type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f"输出目录 (默认: {DEFAULT_OUTPUT_DIR})")
    
    args = parser.parse_args()
    
    collector = HeadPoseDataCollector(
        device_id=args.camera,
        width=args.width,
        height=args.height,
        fps=args.fps,
        output_dir=args.output
    )
    
    collector.duration = args.duration
    
    try:
        print("按数字键开始录制相应类型的数据:")
        print("  0 - 静止 (stationary)")
        print("  1 - 点头 (nodding)")
        print("  2 - 摇头 (shaking)")
        print("  3 - 其他动作 (other)")
        print("按 Q 退出程序")
        
        collector.run()
        
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        collector.shutdown()

if __name__ == "__main__":
    main()
    