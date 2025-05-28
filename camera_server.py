from flask import Flask, Response
import cv2
import time
import threading
from utils.camera_manager import get_camera_manager
import numpy as np

app = Flask(__name__)
camera_manager = get_camera_manager()
frame_lock = threading.Lock()
current_frame = None
display_enabled = True
frame_count = 0
start_time = time.time()

def initialize_camera():
    result = camera_manager.initialize_camera(source=0, width=640, height=480)
    if result != 0:
        raise RuntimeError(f"无法启动相机，错误码: {result}")
    return True

def capture_frames():
    global current_frame, frame_count, start_time
    
    if not camera_manager.is_running():
        initialize_camera()
        
    frame_interval = 1.0 / 60.0
    
    while display_enabled:
        loop_start = time.time()
        
        success, frame = camera_manager.read_frame()
        if success:
            with frame_lock:
                current_frame = frame
                frame_count += 1
                
            elapsed_time = time.time() - start_time
            if elapsed_time >= 1.0:
                fps = frame_count / elapsed_time
                print(f"当前FPS: {fps:.2f}")
                frame_count = 0
                start_time = time.time()
        
        process_time = time.time() - loop_start
        if process_time < frame_interval:
            time.sleep(frame_interval - process_time)

def display_frames():
    global display_enabled
    cv2.namedWindow('Camera Feed', cv2.WINDOW_NORMAL)
    
    while display_enabled:
        with frame_lock:
            if current_frame is not None:
                frame_to_display = current_frame.copy()
            else:
                continue
        
        elapsed_time = time.time() - start_time
        if elapsed_time > 0:
            fps = frame_count / elapsed_time
            cv2.putText(frame_to_display, f"FPS: {fps:.2f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Camera Feed', frame_to_display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            display_enabled = False
            break

def generate_frames_for_web():
    while display_enabled:
        with frame_lock:
            if current_frame is None:
                time.sleep(0.01)
                continue
            frame_to_send = current_frame.copy()
        
        ret, buffer = cv2.imencode('.jpg', frame_to_send)
        if not ret:
            continue
            
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames_for_web(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/image')
def image():
    with frame_lock:
        if current_frame is None:
            return "Error: No frame available.", 503
        frame_to_send = current_frame.copy()
    
    ret, buffer = cv2.imencode('.jpg', frame_to_send)
    if not ret:
        return "Error: Could not encode image.", 500
    
    frame_bytes = buffer.tobytes()
    return Response(frame_bytes, mimetype='image/jpeg')

if __name__ == '__main__':
    initialize_camera()
    
    capture_thread = threading.Thread(target=capture_frames, daemon=True)
    capture_thread.start()
    
    display_thread = threading.Thread(target=display_frames, daemon=True)
    display_thread.start()
    
    try:
        app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
    finally:
        display_enabled = False
        if capture_thread.is_alive():
            capture_thread.join(timeout=1.0)
        if display_thread.is_alive():
            display_thread.join(timeout=1.0)
        cv2.destroyAllWindows()
        camera_manager.release_camera()
