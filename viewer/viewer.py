from flask import Flask, render_template, request, jsonify,redirect, url_for
from individuation import Individuation

import sys
# sys.path.append(r'..')
sys.path.append(r'C:\2025spring\软件工程\小组作业\NKU_SoftwareEngineering')

from applications.application import Application

viewer = Flask(__name__)
# # 初始的 gesture_data 和 text_list
# gesture_data = {
#     "左转": ["选项A", "选项B", "选项C"],
#     "右转": ["选项A", "选项B", "选项C"],
#     "停止": ["选项A", "选项B", "选项C"]
# }

# text_list = {
#     '语音识别': ["开启", "关闭", "自动"],
#     '语音控制': ["开启", "关闭", "自动"]
# }

@viewer.route('/')
def index():
    return render_template('index.html')

# 播放音乐：调用 music_play 接口
def test(music_name):
    print(f"播放音乐：{music_name}")
    Application.schedule(Application.type.music_play, [music_name])

@viewer.route('/play_music', methods=['POST'])
def play_music():
    data = request.get_json()
    music_name = data.get('music')
    test(music_name)
    return '', 204  # No Content

# 暂停/继续：调用 music_change_pause 接口
@viewer.route('/pause_music', methods=['POST'])
def pause_music():
    pause_music_handler()
    return '', 204

def pause_music_handler():
    print("暂停或继续播放音乐")
    Application.schedule(Application.type.music_change_pause, [])

@viewer.route('/music')
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

@viewer.route('/navigation')
def navigation():
    return render_template('navigation.html')

@viewer.route('/status')
def status():
    return render_template('status.html')

# @viewer.route('/update_config', methods=['POST'])
# def update_config():
#     global text_list, gesture_data

#     text_list = {}
#     a = ["开启", "关闭", "自动"]
    

#     # 从请求中获取新的 text_list 和 gesture_data
#     data = request.get_json()
#     x    = Application.get_application_name()
#     text_list = {x[i]: a for i in range(len(x))}
#     print(text_list)
#     # 如果提供了新的 text_list，就更新它
#     if 'text_list' in data:
#         text_list = data['text_list']
    
#     # 如果提供了新的 gesture_data，就更新它
#     if 'gesture_data' in data:
#         gesture_data = data['gesture_data']
    
#     # 返回更新后的配置
#     return jsonify({
#         'status': 'ok',
#         'message': '配置已更新',
#         'text_list': text_list,
#         'gesture_data': gesture_data
#     })

# @viewer.route('/config')
# def config():
#     # 定义一些测试数据，gesture_data 是一个字典
#     gesture_data = {
#         "左转": ["选项A", "选项B", "选项C"],
#         "右转": ["选项A", "选项B", "选项C"],
#         "停止": ["选项A", "选项B", "选项C"]
#     }
    
#     # 定义 text_list 和其对应的选项
#     text_list = {
#         '语音识别': ["开启", "关闭", "自动"],
#         '语音控制': ["开启", "关闭", "自动"]
#     }
    
#     return render_template('config.html', text_list=text_list, gesture_data=gesture_data)
@viewer.route('/config', methods=['GET'])
def config():
    # global text_list, gesture_data

    # text_list = {}
    # a = ["开启", "关闭", "自动"]
    

    # # 从请求中获取新的 text_list 和 gesture_data
    # data = request.get_json()
    # x    = Application.get_application_name()
    # text_list = {x[i]: a for i in range(len(x))}
    # print(text_list)
    # # 如果提供了新的 text_list，就更新它
    # if 'text_list' in data:
    #     text_list = data['text_list']
    
    # # 如果提供了新的 gesture_data，就更新它
    # if 'gesture_data' in data:
    #     gesture_data = data['gesture_data']
    gesture_names = Individuation.get_gesture_names()
    text_list_a = {}
    gesture_data_a = {}
    print("text_list_a:", text_list_a)
    #text_list['语音识别'] = ["开启", "关闭"]
    a = ["开启", "关闭"]
    x    = Application.get_application_name()
    print("x:", x)
    text_list_a = {x[i]: a for i in range(len(x))}
    print("gesture_names:", gesture_names)
    gesture_data_a  = {gesture_names[i]: x for i in range(len(gesture_names))}
    return render_template('config.html', text_list=text_list_a, gesture_data=gesture_data_a)

@viewer.route('/auto')
def auto():
    return render_template('auto.html')

def exopen_music():
    render_template("auto.html", target_url="http://127.0.0.1:5000/music")
    
#轮询

# requests.post('http://127.0.0.1:5000/trigger_action', json={'action': 'music'})

# 后端 Flask 中
last_action = None



# @viewer.route('/trigger_action', methods=['POST'])
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
@viewer.route('/trigger_action', methods=['POST'])
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
    
@viewer.route('/get_action')
def get_action():
    global last_action
    action = last_action
    last_action = None  # 用后清除
    return jsonify({'action': action})

def init_viewer():
    viewer.run(debug=False)



blinking_enabled = True  # 默认开启闪烁

@viewer.route('/set_blinking', methods=['POST'])
def set_blinking():
    global blinking_enabled
    data = request.get_json()
    blinking_enabled = data.get('enabled', True)
    print(f"🔴 闪烁状态设置为: {blinking_enabled}")
    return jsonify({'status': 'ok', 'blinking': blinking_enabled})

@viewer.route('/get_blinking', methods=['GET'])
def get_blinking():
    global blinking_enabled
    return jsonify({'blinking': blinking_enabled})
# Flask 后端
latest_message = "默认警告信息"

@viewer.route('/update_string', methods=['POST'])
def update_string():
    global latest_message
    data = request.get_json()
    latest_message = data.get('message', '无内容')
    print(f"✅ 收到外部消息：{latest_message}")  # ✅ 终端输出确认
    return jsonify({'updated_message': latest_message})

@viewer.route('/get_latest_message', methods=['GET'])
def get_latest_message():
    return jsonify({'updated_message': latest_message})



@viewer.route('/voice', methods=['GET'])
def voice_page():
    # 假设你通过某个逻辑得到了以下测试列表：
    text_list = ["请说出导航目的地", "请说出音乐类型", "请说出车辆状态请求"]
    dropdown_options = ["选项A", "选项B", "选项C"]
    
    return render_template('voice.html', text_list=text_list, dropdown_options=dropdown_options)

@viewer.route('/gesture', methods=['GET'])
def gesture_page():
    # Assume you get the following test list for gestures
    text_list = ["请做出左转手势", "请做出右转手势", "请做出停止手势"]
    dropdown_options = ["选项A", "选项B", "选项C"]
    
    return render_template('gesture.html', text_list=text_list, dropdown_options=dropdown_options)
@viewer.route('/call_void', methods=['POST'])
def call_void():
    data = request.get_json()
    status = data.get('status', '空')
    void(status)
    return '', 204  # 无返回内容

def void(status):
    # 空函数添加参数
    print(f"🚗 收到车辆状态输入：{status}")

if __name__ == '__main__':
    viewer.run(debug=True)
    