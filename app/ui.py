from flask import Flask, render_template, request, jsonify,redirect, url_for

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
    print("🎵 已跳转到 music 页面")
    try:
        music_info = Application.schedule(Application.type.music_getlist, [])
    except Exception as e:
        print(f"❌ 获取音乐列表失败: {e}")
        music_info = []
    print("🎵 已跳转到 music 页面2")
    # render_template("auto.html", target_url="http://127.0.0.1:5000/music")
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

def exopen_music():
    render_template("auto.html", target_url="http://127.0.0.1:5000/music")
    
#轮询

# requests.post('http://127.0.0.1:5000/trigger_action', json={'action': 'music'})

# 后端 Flask 中
last_action = None



# @app.route('/trigger_action', methods=['POST'])
# def trigger_action():
#     global last_action
#     data = request.get_json()
#     action = data.get('action')
#     if action == 'music':
#         print("✅ 收到 POST 请求：music")  # 清晰可见的终端日志
#         last_action = 'music'
#         return jsonify({'status': 'ok', 'message': 'Music action triggered'})
#     else:
#         print("⚠️ 收到未知 action：", action)
#         return jsonify({'status': 'error', 'message': 'Unknown action'}), 400
@app.route('/trigger_action', methods=['POST'])
def trigger_action():
    global last_action
    data = request.get_json()
    action = data.get('action')
    if action in ['music', 'navigation', 'status', 'config', 'auto']:
        last_action = action
        print(f"✅ 收到 POST 请求：{action}")
        return redirect(url_for(action))  # 自动跳转到对应的页面
    else:
        return jsonify({'status': 'error', 'message': 'Unknown action'}), 400
    
@app.route('/get_action')
def get_action():
    global last_action
    action = last_action
    last_action = None  # 用后清除
    return jsonify({'action': action})


if __name__ == '__main__':
    app.run(debug=True)
