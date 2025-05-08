from flask import Flask, render_template, request  # 注意引入 request
import sys
sys.path.append(r'C:/2025spring/软件工程/小组作业/NKU_SoftwareEngineering')
from applications.application import Application

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def test(music_name):
    print(f"播放音乐：{music_name}")
    # 你可以在这里替换为调用播放器或 Application 的方法
    Application.schedule(Application.type.music_play, [music_name])

@app.route('/play_music', methods=['POST'])
def play_music():
    data = request.get_json()
    music_name = data.get('music')
    test(music_name)
    return '', 204  # 无返回内容（No Content）

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
