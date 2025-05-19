import math
import time
import cv2
import mediapipe as mp
import numpy as np
from collections import deque

class FaceDetector:
    def __init__(self):
        # 初始化面部网格检测器
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            max_num_faces=1
        )
        # 全局参数配置
        self.ALPHA = 0.3
        self.focus_history = deque(maxlen=5)
        self.WINDOW_SIZE = 30
        self.DYNAMIC_THRESHOLD_RATIO = 1.5
        self.VERTICAL_SENSITIVITY = 2.0
        self.BASE_NOD_THRESH = 8
        self.BASE_SHAKE_THRESH = 10
        self.ZC_COUNT_THRESH = 2
        self.MOTION_INTERVAL = 1.5
        self.GAZE_SMOOTHING_WINDOW = 25
        self.VERTICAL_SMOOTH_WEIGHTS = np.array([0.6, 0.4])
        # 定义左右眼关键点索引
        self.left_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.right_eye_indices = [263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 384, 385, 386, 387, 388, 398]
        # 3D人脸坐标
        self.face_3d = np.array([
            [285, 528, 200],
            [285, 371, 152],
            [197, 574, 128],
            [173, 425, 108],
            [360, 574, 128],
            [391, 425, 108]
        ], dtype=np.float64)
        # 全局状态变量
        self.angle_history = deque(maxlen=self.WINDOW_SIZE)
        self.pitch_history = deque(maxlen=self.WINDOW_SIZE)
        self.yaw_history = deque(maxlen=self.WINDOW_SIZE)
        self.roll_history = deque(maxlen=self.WINDOW_SIZE)
        self.left_pupil_center_history = deque(maxlen=self.GAZE_SMOOTHING_WINDOW)
        self.right_pupil_center_history = deque(maxlen=self.GAZE_SMOOTHING_WINDOW)
        self.left_gaze_direction_history = deque(maxlen=self.GAZE_SMOOTHING_WINDOW)
        self.right_gaze_direction_history = deque(maxlen=self.GAZE_SMOOTHING_WINDOW)
        self.zero_cross = {
            'pitch': {'count': 0, 'prev_sign': 1, 'timestamps': deque(maxlen=10)},
            'yaw': {'count': 0, 'prev_sign': 1, 'timestamps': deque(maxlen=10)}
        }
        self.last_action_time = 0
        self.detected_action = ""
        self.action_cooldown = 0

    def rotation_matrix_to_angles(self, rotation_matrix):
        """
        将旋转矩阵转换为欧拉角（X, Y, Z轴角度）
        参数:
            rotation_matrix: 3x3旋转矩阵
        返回:
            欧拉角数组（度数）: [pitch, yaw, roll]
        """
        x = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        y = math.atan2(-rotation_matrix[2, 0],
                       math.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2))
        z = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        return np.array([x, y, z]) * 180.0 / math.pi

    def calculate_dynamic_threshold(self, values, base_thresh, is_vertical=False):
        """
        计算动态阈值（基于最近数据的平均幅度）
        参数:
            values: 历史数据队列
            base_thresh: 基础阈值
            is_vertical: 是否垂直方向（影响灵敏度）
        返回:
            动态调整后的阈值
        """
        if len(values) < 5:
            return base_thresh
        recent_values = list(values)[-10:]
        avg_amplitude = np.mean(np.abs(recent_values))
        sensitivity = self.VERTICAL_SENSITIVITY if is_vertical else self.DYNAMIC_THRESHOLD_RATIO
        return max(base_thresh, avg_amplitude * sensitivity)

    def check_zero_cross(self, current_value, axis):
        """
        检测信号过零次数（符号变化）
        参数:
            current_value: 当前信号值
            axis: 检测轴（'pitch'/'yaw'）
        返回:
            当前过零次数
        """
        current_sign = 1 if current_value >= 0 else -1
        state = self.zero_cross[axis]
        if state['prev_sign'] != current_sign:
            state['count'] += 1
            state['timestamps'].append(time.time())
            while state['timestamps']:
                if time.time() - state['timestamps'][0] > self.MOTION_INTERVAL:
                    state['timestamps'].popleft()
                    state['count'] = max(0, state['count'] - 1)
                else:
                    break
        state['prev_sign'] = current_sign
        return state['count']

    def analyze_motion_pattern(self, values, threshold):
        """
        分析运动模式（是否有足够幅度的波动）
        参数:
            values: 角度历史数据
            threshold: 动作检测阈值
        返回:
            是否检测到有效运动模式
        """
        if len(values) < self.WINDOW_SIZE // 2:
            return False
        smoothed = np.convolve(values, np.ones(3) / 3, mode='valid')
        max_idx = np.argmax(smoothed)
        min_idx = np.argmin(smoothed)
        max_val = smoothed[max_idx]
        min_val = smoothed[min_idx]
        if abs(max_val - min_val) > threshold:
            time_diff = abs(max_idx - min_idx)
            if time_diff > len(smoothed) // 4 and (max_val * min_val < 0):
                return True
        return False

    def detect_pupil_center(self, eye_region):
        """
        检测瞳孔中心（基于图像阈值和轮廓检测）
        参数:
            eye_region: 眼部ROI图像
        返回:
            瞳孔中心坐标 (x, y) 或 None
        """
        try:
            gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            min_area = eye_region.shape[0] * eye_region.shape[1] * 0.01
            valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
            if valid_contours:
                largest_contour = max(valid_contours, key=cv2.contourArea)
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        except Exception as e:
            print(f"瞳孔检测错误: {e}")
        return None

    def calculate_eye_center(self, eye_landmarks, image):
        """
        计算眼部区域中心（基于 landmarks 坐标平均）
        参数:
            eye_landmarks: 眼部关键点列表
            image: 原始图像
        返回:
            眼部中心坐标 (x, y)
        """
        x_coords = [int(lm.x * image.shape[1]) for lm in eye_landmarks]
        y_coords = [int(lm.y * image.shape[0]) for lm in eye_landmarks]
        return (int(np.mean(x_coords)), int(np.mean(y_coords)))

    def calculate_gaze_direction(self, pupil_center, eye_center, image_height):
        """
        计算三维视线方向（基于眼球球面模型）
        参数:
            pupil_center: 瞳孔中心坐标
            eye_center: 眼部中心坐标
            image_height: 图像高度（用于深度比例计算）
        返回:
            归一化视线向量 (dx, dy, dz) 或 None
        """
        if not pupil_center or not eye_center:
            return None
        EYE_SPHERE_RADIUS = 15.0
        DEPTH_SCALE = image_height / 300
        ox, oy = eye_center
        px, py = pupil_center
        dx_2d = px - ox
        dy_2d = py - oy
        distance_2d = np.sqrt(dx_2d ** 2 + dy_2d ** 2)
        if distance_2d < 1e-6:
            return (0, 0, 1)
        value = EYE_SPHERE_RADIUS ** 2 - (distance_2d / DEPTH_SCALE) ** 2
        dz = np.sqrt(value) if value >= 0 else 0.1
        norm = np.sqrt(dx_2d ** 2 + dy_2d ** 2 + (dz * DEPTH_SCALE) ** 2)
        if norm < 1e-6:
            return None
        return (dx_2d / norm, dy_2d / norm, (dz * DEPTH_SCALE) / norm)

    def calculate_focus_point(self, left_eye_center, left_gaze, right_eye_center, right_gaze, w, h):
        """
        计算双眼视线交点（焦点）
        参数:
            left/right_eye_center: 左右眼中心坐标
            left/right_gaze: 左右眼视线向量
            w/h: 图像宽度和高度
        返回:
            焦点坐标 (x, y) 或屏幕中心
        """
        if (left_gaze is None or not np.any(left_gaze)) or \
           (right_gaze is None or not np.any(right_gaze)):
            return (w // 2, h // 2)
        a1, b1, c1 = left_gaze
        a2, b2, c2 = right_gaze
        if c1 <= -0.1 or c2 <= -0.1:
            return (w // 2, h // 2)
        x1, y1 = left_eye_center
        x2, y2 = right_eye_center
        denom = a1 * c2 - a2 * c1
        if abs(denom) < 1e-6:
            return (w // 2, h // 2)
        try:
            t1 = (c2 * (x2 - x1) - a2 * (y2 - y1)) / denom
            t2 = (c1 * (x2 - x1) - a1 * (y2 - y1)) / denom
        except ZeroDivisionError:
            return (w // 2, h // 2)
        focus_x = max(0, min(int(x1 + t1 * a1), w))
        focus_y = max(0, min(int(y1 + t1 * b1), h))
        return (focus_x, focus_y)

    def smooth_values(self, history, new_value):
        """
        指数加权移动平均平滑
        参数:
            history: 历史值队列
            new_value: 新值 (x, y)
        返回:
            平滑后的值
        """
        if not history:
            return new_value
        prev_x, prev_y = history[-1]
        curr_x, curr_y = new_value
        return (int(self.ALPHA * curr_x + (1 - self.ALPHA) * prev_x),
                int(self.ALPHA * curr_y + (1 - self.ALPHA) * prev_y))

    def get_eye_region(self, eye_landmarks, image):
        """
        获取眼部ROI区域
        参数:
            eye_landmarks: 眼部关键点列表
            image: 原始图像
        返回:
            眼部区域图像, 左上角偏移量 (x1, y1)
        """
        x_coords = [int(lm.x * image.shape[1]) for lm in eye_landmarks]
        y_coords = [int(lm.y * image.shape[0]) for lm in eye_landmarks]
        x1, y1 = min(x_coords), min(y_coords)
        x2, y2 = max(x_coords), max(y_coords)
        return image[y1:y2, x1:x2], (x1, y1)

    def horizontal_ratio(self, left_pupil, left_x1, left_x2, right_pupil, right_x1, right_x2):
        """
        计算水平方向瞳孔位置比例（0-1）
        参数:
            left/right_pupil: 左右瞳孔中心
            left/right_x1/x2: 左右眼ROI边界
        返回:
            平均水平比例或None（无效时）
        """
        if not left_pupil or not right_pupil:
            return None
        left_ratio = (left_pupil[0] - left_x1) / max(left_x2 - left_x1, 1)
        right_ratio = (right_pupil[0] - right_x1) / max(right_x2 - right_x1, 1)
        return 0.5 * (np.clip(left_ratio, 0, 1) + np.clip(right_ratio, 0, 1))

    def vertical_ratio(self, left_pupil, left_y1, left_y2, right_pupil, right_y1, right_y2):
        """
        计算垂直方向瞳孔位置比例（0-1），带左右一致性检查
        参数:
            left/right_pupil: 左右瞳孔中心
            left/right_y1/y2: 左右眼ROI边界
        返回:
            平滑后垂直比例或None（不一致时）
        """
        if not left_pupil or not right_pupil:
            return None
        left_ratio = (left_pupil[1] - left_y1) / max(left_y2 - left_y1, 1)
        right_ratio = (right_pupil[1] - right_y1) / max(right_y2 - right_y1, 1)
        if abs(left_ratio - right_ratio) > 0.25:
            return None
        return np.clip(np.dot([left_ratio, right_ratio], self.VERTICAL_SMOOTH_WEIGHTS), 0, 1)

    def get_focus_region(self, h_ratio, v_ratio, focus_point, w, h):
        """
        确定焦点所在屏幕区域
        参数:
            h_ratio: 水平比例（0-1）
            v_ratio: 垂直比例（0-1）
            focus_point: 焦点坐标
            w/h: 图像尺寸
        返回:
            区域名称（如"Top Left"）
        """
        if None in [h_ratio, v_ratio, focus_point]:
            return "Unknown"
        x_region = "Center"
        if h_ratio < 0.4:
            x_region = "Right"
        elif h_ratio > 0.6:
            x_region = "Left"
        y_region = "Center"
        if v_ratio < 0.4:
            y_region = "Top"
        elif v_ratio > 0.55:
            y_region = "Bottom"
        return f"{y_region} {x_region}".strip()

    def process_face(self, image):
        """
        主处理函数：处理单帧图像，检测面部姿态、眼球运动和动作
        参数:
            image: 输入BGR图像
        返回:
            处理后的可视化图像
        """
        # 转换为RGB格式供MediaPipe处理
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image)
        # 转换回BGR格式用于OpenCV显示
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        h, w, _ = image.shape
        face_2d = []

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # 提取特定关键点用于姿态估计
                for idx in [1, 9, 57, 130, 287, 359]:
                    lm = face_landmarks.landmark[idx]
                    face_2d.append([int(lm.x * w), int(lm.y * h)])
                face_2d = np.array(face_2d, dtype=np.float64)
                cam_matrix = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64)
                dist_matrix = np.zeros((4, 1), dtype=np.float64)
                # 求解3D到2D投影，获取旋转和平移向量
                success, rvec, tvec = cv2.solvePnP(self.face_3d, face_2d, cam_matrix, dist_matrix)
                rotation_matrix, _ = cv2.Rodrigues(rvec)
                # 转换为欧拉角
                euler_angles = self.rotation_matrix_to_angles(rotation_matrix)
                pitch, yaw, roll = euler_angles
                self.pitch_history.append(pitch)
                self.yaw_history.append(yaw)
                self.roll_history.append(roll)
                self.angle_history.append({'pitch': pitch, 'yaw': yaw, 'time': time.time()})

                # 计算平滑角度
                smooth_pitch = np.mean(list(self.pitch_history)[-5:]) if self.pitch_history else 0
                smooth_yaw = np.mean(list(self.yaw_history)[-5:]) if self.yaw_history else 0
                # 计算动态阈值
                dyn_nod_thresh = self.calculate_dynamic_threshold(self.pitch_history, self.BASE_NOD_THRESH, is_vertical=True)
                dyn_shake_thresh = self.calculate_dynamic_threshold(self.yaw_history, self.BASE_SHAKE_THRESH)
                current_time = time.time()
                time_since_last = current_time - self.last_action_time

                # 检测点头动作
                pitch_zc = self.check_zero_cross(smooth_pitch, 'pitch')
                if len(self.pitch_history) >= self.WINDOW_SIZE and abs(smooth_pitch) < 20 and \
                        self.analyze_motion_pattern(list(self.pitch_history), dyn_nod_thresh) and \
                        pitch_zc >= self.ZC_COUNT_THRESH and time_since_last > self.MOTION_INTERVAL and \
                        np.ptp(self.yaw_history) < 20 and \
                        np.ptp(self.roll_history) < 8:
                    self.detected_action = "NOD"
                    self.last_action_time = current_time
                    self.zero_cross['pitch']['count'] = 0

                # 检测摇头动作
                yaw_zc = self.check_zero_cross(smooth_yaw, 'yaw')
                if len(self.yaw_history) >= self.WINDOW_SIZE and \
                        self.analyze_motion_pattern(list(self.yaw_history), dyn_shake_thresh) and \
                        yaw_zc >= self.ZC_COUNT_THRESH and time_since_last > self.MOTION_INTERVAL and \
                        np.ptp(self.pitch_history) < 15 and \
                        np.ptp(self.roll_history) < 20 and \
                        np.ptp(self.roll_history) > 8:
                    self.detected_action = "SHAKE"
                    self.last_action_time = current_time
                    self.zero_cross['yaw']['count'] = 0

                # 获取左右眼关键点
                left_eye_landmarks = [face_landmarks.landmark[idx] for idx in self.left_eye_indices]
                right_eye_landmarks = [face_landmarks.landmark[idx] for idx in self.right_eye_indices]
                # 计算眼中心
                left_eye_center = self.calculate_eye_center(left_eye_landmarks, image)
                right_eye_center = self.calculate_eye_center(right_eye_landmarks, image)
                # 获取眼ROI
                left_roi, left_offset = self.get_eye_region(left_eye_landmarks, image)
                right_roi, right_offset = self.get_eye_region(right_eye_landmarks, image)

                # 检测瞳孔中心
                left_pupil = self.detect_pupil_center(left_roi)
                if not left_pupil and self.left_pupil_center_history:
                    recent = list(self.left_pupil_center_history)[-3:]
                    if len(recent) >= 2:
                        left_pupil = (int(np.mean([p[0] for p in recent])), int(np.mean([p[1] for p in recent])))
                if left_pupil:
                    global_left_pupil = (left_pupil[0] + left_offset[0], left_pupil[1] + left_offset[1])
                    smoothed_left = self.smooth_values(self.left_pupil_center_history, global_left_pupil)
                    self.left_pupil_center_history.append(smoothed_left)

                right_pupil = self.detect_pupil_center(right_roi)
                if not right_pupil and self.right_pupil_center_history:
                    recent = list(self.right_pupil_center_history)[-3:]
                    if len(recent) >= 2:
                        right_pupil = (int(np.mean([p[0] for p in recent])), int(np.mean([p[1] for p in recent])))
                if right_pupil:
                    global_right_pupil = (right_pupil[0] + right_offset[0], right_pupil[1] + right_offset[1])
                    smoothed_right = self.smooth_values(self.right_pupil_center_history, global_right_pupil)
                    self.right_pupil_center_history.append(smoothed_right)

                # 获取平滑后瞳孔中心
                smoothed_left_pupil = self.left_pupil_center_history[-1] if self.left_pupil_center_history else None
                smoothed_right_pupil = self.right_pupil_center_history[-1] if self.right_pupil_center_history else None
                # 计算视线方向
                left_gaze = self.calculate_gaze_direction(smoothed_left_pupil, left_eye_center, h)
                right_gaze = self.calculate_gaze_direction(smoothed_right_pupil, right_eye_center, h)
                if left_gaze:
                    self.left_gaze_direction_history.append(left_gaze)
                if right_gaze:
                    self.right_gaze_direction_history.append(right_gaze)
                # 平滑视线方向
                smoothed_left_gaze = np.mean(self.left_gaze_direction_history, axis=0) if self.left_gaze_direction_history else None
                smoothed_right_gaze = np.mean(self.right_gaze_direction_history, axis=0) if self.right_gaze_direction_history else None
                # 计算焦点
                focus_point = self.calculate_focus_point(
                    left_eye_center, smoothed_left_gaze,
                    right_eye_center, smoothed_right_gaze, w, h
                )
                # 计算瞳孔位置比例
                h_ratio = self.horizontal_ratio(
                    smoothed_left_pupil, left_offset[0], left_offset[0] + left_roi.shape[1],
                    smoothed_right_pupil, right_offset[0], right_offset[0] + right_roi.shape[1]
                )
                v_ratio = self.vertical_ratio(
                    smoothed_left_pupil, left_offset[1], left_offset[1] + left_roi.shape[0],
                    smoothed_right_pupil, right_offset[1], right_offset[1] + right_roi.shape[0]
                )

                if focus_point and h_ratio and v_ratio:
                    # 确定焦点区域并显示
                    focus_region = self.get_focus_region(h_ratio, v_ratio, focus_point, w, h)
                    cv2.putText(image, f"focus region: {focus_region}", (10, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 0, 200), 2)

                # 显示角度信息
                y_pos = 30
                for angle_name, angle_value in zip(['Pitch', 'Yaw', 'Roll'], [smooth_pitch, smooth_yaw, roll]):
                    cv2.putText(image, f"{angle_name}: {angle_value:+.1f}°", (10, y_pos),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 0, 200), 2)
                    y_pos += 30

                if time_since_last < self.MOTION_INTERVAL:
                    # 显示检测到的动作
                    text = f"{self.detected_action}"
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
                    cv2.putText(image, text, (w // 2 - text_size[0] // 2, h // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

                # 绘制瞳孔和视线
                for pupil, color in [(smoothed_left_pupil, (255, 0, 0)), (smoothed_right_pupil, (0, 0, 255))]:
                    if pupil:
                        cv2.circle(image, pupil, 3, color, -1)
                for eye_center, gaze, color in [(left_eye_center, smoothed_left_gaze, (0, 255, 0)),
                                                (right_eye_center, smoothed_right_gaze, (0, 255, 0))]:
                    if eye_center and gaze is not None and np.any(gaze):
                        end_point = (
                            eye_center[0] + int(gaze[0] * 50),
                            eye_center[1] + int(gaze[1] * 50)
                        )
                        cv2.arrowedLine(image, eye_center, end_point, color, 2)

                if focus_point:
                    # 绘制焦点
                    cv2.circle(image, focus_point, 5, (0, 0, 255), -1)

                # 显示额外信息
                info = [
                    f"dyn_nod_thresh {dyn_nod_thresh:.1f}° / dyn_shake_thresh {dyn_shake_thresh:.1f}°",
                    f"zc: P{pitch_zc} Y{yaw_zc}",
                    f"last_detected_action {self.detected_action} ({max(0, self.MOTION_INTERVAL - time_since_last):.1f}s)"
                ]
                y_pos = 380
                for line in info:
                    cv2.putText(image, line, (10, y_pos),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
                    y_pos += 30

        return image

if __name__ == "__main__":
    detector = FaceDetector()
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("读取摄像头帧失败")
            continue
        processed_image = detector.process_face(image)
        cv2.imshow('', processed_image)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    