from flask import Flask, Response
import cv2

app = Flask(__name__)
camera = None

def generate_frames():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError("Could not start camera.")

    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/image')
def image():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        return "Error: Could not start camera.", 500

    success, frame = camera.read()
    if not success:
        return "Error: Could not read frame.", 500
    
    ret, buffer = cv2.imencode('.jpg', frame)
    if not ret:
        return "Error: Could not encode image.", 500
    
    frame_bytes = buffer.tobytes()
    return Response(frame_bytes, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
