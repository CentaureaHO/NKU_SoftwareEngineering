from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np


class VisualizationUtil:

    @staticmethod
    def show_frame(window_name: str, frame: np.ndarray) -> None:
        cv2.imshow(window_name, frame)

    @staticmethod
    def add_text_to_frame(frame: np.ndarray, text: str, position: tuple,
                          color: tuple = (0, 255, 0), font_scale: float = 0.6,
                          thickness: int = 2) -> np.ndarray:
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, color, thickness)
        return frame

    @staticmethod
    def add_detection_info(frame: np.ndarray, detections: Dict[str, Any],
                           position: tuple = (10, 30), line_height: int = 25) -> np.ndarray:
        x, y = position

        for key, value in detections.items():
            if isinstance(value, dict):
                text = f"{key}:"
                VisualizationUtil.add_text_to_frame(frame, text, (x, y))
                y += line_height

                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, list) and len(sub_value) > 5:
                        text = f"  {sub_key}: [...]"
                    else:
                        text = f"  {sub_key}: {sub_value}"

                    VisualizationUtil.add_text_to_frame(
                        frame, text, (x + 10, y))
                    y += line_height
            else:
                text = f"{key}: {value}"
                VisualizationUtil.add_text_to_frame(frame, text, (x, y))
                y += line_height

        return frame

    @staticmethod
    def create_dashboard(frames: List[np.ndarray], titles: List[str],
                         grid_size: tuple = None) -> np.ndarray:
        if not frames:
            return None

        n_frames = len(frames)

        if grid_size is None:
            cols = min(3, n_frames)
            rows = (n_frames + cols - 1) // cols
        else:
            rows, cols = grid_size

        frame_height, frame_width = frames[0].shape[:2]

        dashboard = np.zeros(
            (frame_height * rows, frame_width * cols, 3), dtype=np.uint8)

        for i, (frame, title) in enumerate(zip(frames, titles)):
            if i >= rows * cols:
                break

            row = i // cols
            col = i % cols

            y_start = row * frame_height
            y_end = (row + 1) * frame_height
            x_start = col * frame_width
            x_end = (col + 1) * frame_width

            if len(frame.shape) == 2 or frame.shape[2] == 1:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            dashboard[y_start:y_end, x_start:x_end] = frame

            VisualizationUtil.add_text_to_frame(dashboard, title,
                                                (x_start + 10, y_start + 25),
                                                (0, 0, 255), 0.8)

        return dashboard

    @staticmethod
    def wait_key(delay: int = 1) -> int:
        return cv2.waitKey(delay)

    @staticmethod
    def destroy_windows() -> None:
        cv2.destroyAllWindows()
