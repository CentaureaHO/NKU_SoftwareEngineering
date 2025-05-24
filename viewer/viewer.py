#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#_author = 'Fangyi Liu'

"""
实现后端和后端路由控制
"""
from flask import Flask, render_template, request, jsonify,redirect, url_for

import sys
sys.path.append(r'C:\Users\13033\Desktop\软工大作业5.19.14.00')
#sys.path.append(r'C:\2025spring\软件工程\小组作业\NKU_SoftwareEngineering')
from applications.application import Application
from individuation import individuation
import json
import os
import time
import requests # Import the requests library

viewer = Flask(__name__)

# Your Amap Web Service API Key
# Make sure this Key has permissions for Geocoding and Driving Route Planning
AMAP_WEB_SERVICE_KEY = "3479b328113102a27c20e818c3bc143c"

# Base URLs for Amap Web Service APIs
GEOCODE_API_URL = "https://restapi.amap.com/v3/geocode/geo"
DRIVING_API_URL = "https://restapi.amap.com/v3/direction/driving"

def get_coords(address):
    """Helper function to get coordinates from address using Amap Geocoding API."""
    params = {
        'address': address,
        'key': AMAP_WEB_SERVICE_KEY
    }
    try:
        response = requests.get(GEOCODE_API_URL, params=params)
        response.raise_for_status() # Raise an exception for bad status codes
        result = response.json()
        if result and result['status'] == '1' and result['geocodes']:
            # Return the location string (longitude,latitude)
            return result['geocodes'][0]['location']
        else:
            print(f"Geocoding failed for {address}: {result.get('info', 'Unknown error')}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error calling Geocoding API for {address}: {e}")
        return None

# 渲染索引页
@viewer.route('/')
def index():
    return render_template('index.html')

# 渲染音乐页面
@viewer.route('/music')
def music():
    print("🎵 已跳转到 music 页面")
    try:
        music_info = Application.schedule(Application.type.music_getlist, [])
    except Exception as e:
        print(f"❌ 获取音乐列表失败: {e}")
        music_info = []
    print("🎵 已跳转到 music 页面2")
    return render_template('music.html', music_info=music_info)

# 渲染导航页面
@viewer.route('/navigation')
def navigation():
    print("已跳转到导航页面")
    info = Application.schedule(Application.type.navigation_getlist, [])
    return render_template('navigation.html', info=info)

# 渲染车辆状态监测页面
@viewer.route('/status')
def status():
    status_info = Application.schedule(Application.type.vehicle_state, [])
    oil_quantity = status_info[0]
    tire_pressure = status_info[1]
    mileage = status_info[2]
    #print("status_info:", status_info)
    return render_template('status.html', oil_quantity = oil_quantity,tire_pressure = tire_pressure,mileage = mileage)

# 渲染个性化配置页面
@viewer.route('/config', methods=['GET'])
def config():
    # 获取手势名称
    gesture_names = individuation.get_gesture_names()
    # print("gesture_names:", gesture_names)
    # 获取应用功能名称
    application_names = Application.get_application_names()
    # print("application_names:", application_names)
    # 根据应用程序名称设置 text_list
    text_list = {application_names[i]: [] for i in range(len(application_names))}
    print("text_list:", text_list)
    # 根据应用功能和手势名称设置 gesture_data
    gesture_data = {gesture_names[i]: application_names for i in range(len(gesture_names))}
    print("gesture_data:", gesture_data)
    # 返回页面并渲染配置
    return render_template('config.html', text_list=text_list, gesture_data=gesture_data)

# 渲染权限设置页面
@viewer.route('/settings')
def settings():
    print(" 已跳转到权限设置页面")
    try:
        from multimodal_controller import setting
        music_info = setting.get_voiceprints()
        driver_info = setting.get_driver()
    except Exception as e:
        print(f"获取声纹列表/驾驶员失败: {e}")
        music_info = []
        driver_info = None

    if not driver_info:
        driver_info = "无"
    print("music_info:", music_info)
    print("driver_info:", driver_info)
    return render_template('settings.html', music_info=music_info, driver_info=driver_info)

# 播放音乐
@viewer.route('/play_music', methods=['POST'])
def play_music():
    data = request.get_json()
    music_name = data.get('music')
    print(f"播放音乐：{music_name}")
    Application.schedule(Application.type.music_play, [music_name])
    return '', 204  # No Content

# 暂停/继续播放音乐
@viewer.route('/pause_music', methods=['POST'])
def pause_music():
    print("暂停或继续播放音乐")
    Application.schedule(Application.type.music_change_pause, [])
    return '', 204

# 导航
@viewer.route('/call_navigate', methods=['POST'])
def navigate():
    data = request.get_json()
    start_location = data.get('start')
    end_location = data.get('end')

    if not start_location or not end_location:
        return jsonify({"error": "Please provide start and end locations"}), 400 # Bad Request

    # Get coordinates for start and end locations
    start_coords = get_coords(start_location)
    end_coords = get_coords(end_location)

    if not start_coords:
        return jsonify({"error": f"Could not geocode start location: {start_location}"}), 404
    if not end_coords:
        return jsonify({"error": f"Could not geocode end location: {end_location}"}), 404

    # Call Amap Driving Route Planning API
    driving_params = {
        'origin': start_coords,
        'destination': end_coords,
        'key': AMAP_WEB_SERVICE_KEY,
        'extensions': 'all' # Request detailed route information
    }

    try:
        driving_response = requests.get(DRIVING_API_URL, params=driving_params)
        driving_response.raise_for_status()
        driving_result = driving_response.json()

        if driving_result and driving_result['status'] == '1' and driving_result['route']:
            # Return the route data to the frontend
            return jsonify(driving_result)
        else:
            error_info = driving_result.get('info', 'Unknown route planning error')
            print(f"Driving route planning failed: {error_info}")
            return jsonify({"error": f"Driving route planning failed: {error_info}"}), 500 # Internal Server Error

    except requests.exceptions.RequestException as e:
        print(f"Error calling Driving API: {e}")
        return jsonify({"error": f"Error calling Driving API: {e}"}), 500

@viewer.route('/save_config', methods=['POST'])
def save_config():
    data = request.get_json()
    
    # 获取语音输入框的内容
    voice_inputs = data.get('voiceInputs', {})
    print("保存的语音功能配置:")
    for key, value in voice_inputs.items():
        print(f"{key}: {value}")
    individuation.set_speech_individuation(voice_inputs)

    # 获取手势功能的选择
    gesture_options = data.get('gestureOptions', {})
    print("保存的手势功能配置:")
    for key, value in gesture_options.items():
        print(f"{key}: {value}")
    individuation.set_gesture_individuation(gesture_options)

    return jsonify({'status': 'ok', 'message': '配置已保存'})

@viewer.route('/trigger_action', methods=['POST'])
def trigger_action():
    data = request.get_json()
    action = data.get('action')
    if action in ['music', 'navigation', 'status', 'config', 'auto']:
        print(f"✅ 收到 POST 请求：{action}")
        return redirect(url_for(action))  # 自动跳转到对应的页面
    else:
        return jsonify({'status': 'error', 'message': 'Unknown action'}), 400





# @viewer.route('/auto')
# def auto():
#     return render_template('auto.html')

def exopen_music():
    render_template("auto.html", target_url="http://127.0.0.1:5000/music")
    
#轮询

# requests.post('http://127.0.0.1:5000/trigger_action', json={'action': 'music'})

# 后端 Flask 中
last_action = None



@viewer.route('/get_action')
def get_action():
    global last_action
    action = last_action
    last_action = None  # 用后清除
    return jsonify({'action': action})

def init_viewer():
    viewer.run(debug=False)



blinking_enabled = False  # 默认开启闪烁

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
    print("调用了call_void")
    data = request.get_json()
    status = data.get('status', '空')
    enter_voiceprint(status)
    return '', 204  # 无返回内容

def enter_voiceprint(username):
    print(f"开始录入声纹,用户名为\"{username}\"")
    #from multimodal_controller import controller
    #controller.work_flag = False
    from multimodal_controller import setting
    setting.register_voiceprint(username)
    setting.view_registered_voiceprints()

@viewer.route('/call_delete_user', methods=['POST'])
def delete_user():
    data = request.get_json()
    username = data.get('username', '空')
    print(f"删除用户,用户名为\"{username}\"")
    from multimodal_controller import setting
    setting.delete_voiceprint(username)
    return '', 204

@viewer.route('/call_set_driver', methods=['POST'])
def set_driver():
    data = request.get_json()
    driver_name = data.get('drivername', None)
    print(f"设置驾驶员,用户名为\"{driver_name}\"")
    from multimodal_controller import setting
    setting.set_driver(driver_name)
    return '', 204

if __name__ == '__main__':
    viewer.run(debug=True)
    