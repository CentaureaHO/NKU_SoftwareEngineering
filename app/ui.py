from flask import Flask, render_template, request
import sys
sys.path.append(r'C:/2025spring/软件工程/小组作业/NKU_SoftwareEngineering')
from applications.application import Application

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

# 播放音乐：调用 music_play 接口
def test(music_name):
    print(f"播放音乐：{music_name}")
    Application.schedule(Application.type.music_play, [music_name])

@app.route('/play_music', methods=['POST'])
def play_music():
    data = request.get_json()
    music_name = data.get('music')
    test(music_name)
    return '', 204  # No Content

# 暂停/继续：调用 music_change_pause 接口
@app.route('/pause_music', methods=['POST'])
def pause_music():
    pause_music_handler()
    return '', 204

def pause_music_handler():
    print("暂停或继续播放音乐")
    Application.schedule(Application.type.music_change_pause, [])

@app.route('/music')
def music():
    music_info = Application.schedule(Application.type.music_getlist, [])
    return render_template('music.html', music_info=music_info)

@app.route('/navigation')
def navigation():
    return render_template('navigation.html')

@app.route('/status')
def status():
    return render_template('status.html')

@app.route('/config')
def config():
    return render_template('config.html')

@app.route('/auto')
def auto():
    return render_template('auto.html')

if __name__ == '__main__':
    app.run(debug=True)
