import os
import cv2
import numpy as np
import tensorflow as tf
import pickle
import json
import time
import argparse
from collections import deque
import matplotlib.pyplot as plt

import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
DEFAULT_CAMERA_ID = 0
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480

GESTURE_MAPPING = {
    0: "stationary",
    1: "nodding",
    2: "shaking",
    3: "other"
}

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


class HeadGestureRecognizer:
    def __init__(self, model_dir=DEFAULT_MODEL_DIR, camera_id=DEFAULT_CAMERA_ID, 
                 width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT):
        self.model_dir = model_dir
        self.camera_id = camera_id
        self.width = width
        self.height = height

        self.model_path, self.scaler_path, self.config_path = self._find_latest_model()

        self.model = None
        self.scaler = None
        self.config = None
        self.window_size = None
        self.feature_dim = None

        self.face_mesh = None

        self.face_3d_coords = np.array([
            [285, 528, 200], 
            [285, 371, 152], 
            [197, 574, 128], 
            [173, 425, 108], 
            [360, 574, 128], 
            [391, 425, 108] 
        ], dtype=np.float64)

        self.features_queue = None

        self.cap = None

    def _find_latest_model(self):
        model_files = [f for f in os.listdir(self.model_dir) if f.endswith('.h5')]
        scaler_files = [f for f in os.listdir(self.model_dir) if f.endswith('.pkl')]
        config_files = [f for f in os.listdir(self.model_dir) if f.endswith('.json') and 'model_config' in f]

        if not model_files or not scaler_files or not config_files:
            raise FileNotFoundError(f"找不到完整的模型文件组合，请确保模型目录 {self.model_dir} 中包含 .h5、.pkl 和 config .json 文件")

        model_files.sort(reverse=True)
        scaler_files.sort(reverse=True)
        config_files.sort(reverse=True)

        model_path = os.path.join(self.model_dir, model_files[0])
        scaler_path = os.path.join(self.model_dir, scaler_files[0])
        config_path = os.path.join(self.model_dir, config_files[0])

        return model_path, scaler_path, config_path

    def initialize(self):
        try:
            print(f"正在加载模型文件: {self.model_path}")
            self.model = tf.keras.models.load_model(self.model_path)

            print(f"正在加载标准化器: {self.scaler_path}")
            with open(self.scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)

            print(f"正在加载模型配置: {self.config_path}")
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)

            self.window_size = self.config["window_size"]
            self.feature_dim = self.config["feature_dim"]

            self.features_queue = deque(maxlen=self.window_size)

            self.face_mesh = mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )

            self.cap = cv2.VideoCapture(self.camera_id)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

            if not self.cap.isOpened():
                raise RuntimeError(f"无法打开摄像头 {self.camera_id}")

            print(f"模型初始化完成，窗口大小: {self.window_size}，特征维度: {self.feature_dim}")
            return True

        except Exception as e:
            print(f"初始化失败: {str(e)}")
            return False

    def _rotation_matrix_to_angles(self, rotation_matrix):
        import math
        x = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        y = math.atan2(-rotation_matrix[2, 0], math.sqrt(rotation_matrix[0, 0] ** 2 +
                                                     rotation_matrix[1, 0] ** 2))
        z = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        return np.array([x, y, z]) * 180. / math.pi

    def _euclidean_dist(self, point1, point2):
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

    def _extract_features(self, frame):
        h, w, _ = frame.shape
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)

        if not results.multi_face_landmarks:
            return None

        face_landmarks = results.multi_face_landmarks[0]
        face_coordination_in_image = []

        all_face_coords = []

        nose_point = None
        chin_point = None
        left_ear_point = None
        left_face_point = None
        right_ear_point = None
        right_face_point = None

        for idx, landmark in enumerate(face_landmarks.landmark):
            x, y = int(landmark.x * w), int(landmark.y * h)

            all_face_coords.append((x, y))

            if idx in [1, 9, 57, 130, 287, 359]:
                face_coordination_in_image.append([x, y])

            if idx == HeadPoseParams.LANDMARK_NOSE:
                nose_point = (x, y)
            elif idx == HeadPoseParams.LANDMARK_CHIN:
                chin_point = (x, y)
            elif idx == HeadPoseParams.LANDMARK_LEFT_EAR:
                left_ear_point = (x, y)
            elif idx == HeadPoseParams.LANDMARK_LEFT_FACE:
                left_face_point = (x, y)
            elif idx == HeadPoseParams.LANDMARK_RIGHT_EAR:
                right_ear_point = (x, y)
            elif idx == HeadPoseParams.LANDMARK_RIGHT_FACE:
                right_face_point = (x, y)

        if len(face_coordination_in_image) != 6 or None in (nose_point, chin_point, left_ear_point, left_face_point, right_ear_point, right_face_point):
            return None

        all_face_coords = np.array(all_face_coords)
        x_min, y_min = np.min(all_face_coords, axis=0)
        x_max, y_max = np.max(all_face_coords, axis=0)
        box_width = max(x_max - x_min, 1)
        box_height = max(y_max - y_min, 1)
        box_diagonal = np.sqrt(box_width**2 + box_height**2)
            
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

            if not success:
                return None

            rotation_matrix, _ = cv2.Rodrigues(rotation_vec)
            angles = self._rotation_matrix_to_angles(rotation_matrix)

            pitch = float(angles[0])
            yaw = float(angles[1])
            roll = float(angles[2])

            nose_chin_distance = self._euclidean_dist(nose_point, chin_point)
            left_cheek_width = self._euclidean_dist(left_ear_point, left_face_point)
            right_cheek_width = self._euclidean_dist(right_ear_point, right_face_point)

            basic_features = np.array([
                pitch,
                yaw,
                roll,
                nose_chin_distance,
                left_cheek_width,
                right_cheek_width
            ], dtype=np.float32)

            aspect_ratio = box_width / box_height

            normalized_features = np.array([
                aspect_ratio,
                box_width / box_diagonal,
                box_height / box_diagonal,

                pitch / box_diagonal,
                yaw / box_diagonal,
                roll / box_diagonal,
                
                nose_chin_distance / box_height,
                left_cheek_width / box_width,
                right_cheek_width / box_width
            ], dtype=np.float32)
            
            combined_features = np.concatenate([basic_features, normalized_features])
            return combined_features

        except Exception as e:
            print(f"计算头部姿态时出错: {str(e)}")
            return None
            
    def predict_gesture(self, frame):
        features = self._extract_features(frame)
        if features is None:
            return None
        
        expected_feature_count = self.feature_dim // 2
        if len(features) != expected_feature_count:
            if len(features) > expected_feature_count:
                features = features[:expected_feature_count]
            else:
                padding = np.zeros(expected_feature_count - len(features), dtype=np.float32)
                features = np.concatenate([features, padding])

        self.features_queue.append(features)

        if len(self.features_queue) < self.window_size:
            return None

        diff_seq = []
        features_list = list(self.features_queue)

        for i in range(1, len(features_list)):
            combined_features = np.concatenate([
                features_list[i], 
                features_list[i] - features_list[i-1]
            ])
            diff_seq.append(combined_features)

        sequence = np.array([diff_seq])

        shape = sequence.shape
        sequence_flat = sequence.reshape((-1, self.feature_dim))
        sequence_flat = self.scaler.transform(sequence_flat)
        sequence = sequence_flat.reshape(shape)

        prediction = self.model.predict(sequence, verbose=0)
        gesture_idx = np.argmax(prediction[0])
        confidence = float(prediction[0][gesture_idx])

        return GESTURE_MAPPING[gesture_idx], confidence

    def run(self):
        if not self.cap or not self.cap.isOpened():
            print("摄像头未初始化")
            return

        color_map = {
            "stationary": (0, 255, 0),
            "nodding": (0, 165, 255),
            "shaking": (0, 0, 255),
            "other": (255, 0, 255)
        }

        print("\n按 'q' 键退出程序")
        print("按 'c' 键清空特征队列\n")

        prev_frame_time = 0
        predictions_history = []
        confidences_history = []
        max_history = 50

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            current_frame_time = time.time()
            fps = 1 / (current_frame_time - prev_frame_time) if prev_frame_time > 0 else 0
            prev_frame_time = current_frame_time

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(image_rgb)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        face_landmarks,
                        mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                    )

            result = self.predict_gesture(frame)

            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if result:
                gesture, confidence = result

                predictions_history.append(gesture)
                confidences_history.append(confidence)
                if len(predictions_history) > max_history:
                    predictions_history.pop(0)
                    confidences_history.pop(0)

                color = color_map.get(gesture, (200, 200, 200))
                cv2.putText(frame, f"Gesture: {gesture} ({confidence:.2f})", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            else:
                cv2.putText(frame, "No face detected", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if predictions_history:
                history_img = self._create_prediction_history_image(predictions_history, confidences_history)
                small_history = cv2.resize(history_img, (320, 120))

                x_offset = frame.shape[1] - small_history.shape[1] - 10
                y_offset = 10
                frame[y_offset:y_offset+small_history.shape[0], x_offset:x_offset+small_history.shape[1]] = small_history

            cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, "Press 'c' to clear history", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow("Head Gesture Recognition", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.features_queue.clear()
                predictions_history.clear()
                confidences_history.clear()
                print("特征队列已清空")

        self.cap.release()
        cv2.destroyAllWindows()

    def _create_prediction_history_image(self, predictions, confidences):
        img_height = 120
        img_width = 320
        img = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255

        if not predictions:
            return img

        history_length = len(predictions)
        segment_width = (img_width - 20) / history_length

        color_map = {
            "stationary": (0, 255, 0),
            "nodding": (0, 165, 255),
            "shaking": (0, 0, 255),
            "other": (255, 0, 255)
        }

        cv2.putText(img, "Prediction History", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        for i, (pred, conf) in enumerate(zip(predictions, confidences)):
            x = int(10 + i * segment_width)
            color = color_map.get(pred, (200, 200, 200))
            bar_height = int(conf * 60)
            y_start = img_height - 30 - bar_height
            cv2.rectangle(img, (x, y_start), (x + int(segment_width) - 1, img_height - 30), color, -1)

        legend_y = 40
        for i, (gesture, color) in enumerate(color_map.items()):
            x = 10 + i * 75
            cv2.rectangle(img, (x, legend_y), (x+10, legend_y+10), color, -1)
            cv2.putText(img, gesture, (x+15, legend_y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        return img

    def shutdown(self):
        if self.cap:
            self.cap.release()
        if self.face_mesh:
            self.face_mesh.close()
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="实时头部姿态识别")
    parser.add_argument("--model_dir", type=str, default=DEFAULT_MODEL_DIR,
                       help=f"模型目录 (默认: {DEFAULT_MODEL_DIR})")
    parser.add_argument("--camera", type=int, default=DEFAULT_CAMERA_ID,
                       help=f"摄像头ID (默认: {DEFAULT_CAMERA_ID})")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH,
                       help=f"视频宽度 (默认: {DEFAULT_WIDTH})")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT,
                       help=f"视频高度 (默认: {DEFAULT_HEIGHT})")

    args = parser.parse_args()

    recognizer = HeadGestureRecognizer(
        model_dir=args.model_dir,
        camera_id=args.camera,
        width=args.width,
        height=args.height
    )

    try:
        if not recognizer.initialize():
            print("初始化失败")
            return

        recognizer.run()

    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        recognizer.shutdown()

if __name__ == "__main__":
    main()
