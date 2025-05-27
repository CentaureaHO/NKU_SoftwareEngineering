#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#_author = 'Fangyi Liu'

"""
实现后端和后端路由控制
"""
import threading
from flask import Flask, render_template, request, jsonify,redirect, url_for,Response
import cv2

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import json
import os
import time
import requests # Import the requests library
from utils.camera_manager import get_camera_manager
import numpy as np

viewer = Flask(__name__)
camera_mgr = None
application = None
controller = None

# 顶部添加组件获取函数
def get_application():
    """获取应用程序控制器"""
    from system_init import get_component
    return get_component('application')

def get_controller():
    """获取多模态控制器"""
    from system_init import get_component
    return get_component('controller')

def get_setting():
    """获取设置模块"""
    from system_init import get_component
    return get_component('setting')

def get_individuation():
    """获取设置模块"""
    from system_init import get_component
    return get_component('individuation')

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
    #from applications.application import application
    print("已跳转到 music 页面")
    try:
        application = get_application()
        music_info = application.schedule(application.type.music_getlist, [])
    except Exception as e:
        print(f"获取音乐列表失败: {e}")
        music_info = []
    return render_template('music.html', music_info=music_info)

# 渲染导航页面
@viewer.route('/navigation')
def navigation():
    print("已跳转到导航页面")
    return render_template('navigation.html')

# 渲染车辆状态监测页面
@viewer.route('/status')
def status():
    # from applications.application import application
    application = get_application()
    status_info = application.schedule(application.type.monitor_getlist, [])
    oil_quantity = status_info[0]
    tire_pressure = status_info[1]
    mileage = status_info[2]
    #print("status_info:", status_info)
    return render_template('status.html', oil_quantity = oil_quantity,tire_pressure = tire_pressure,mileage = mileage)

# 渲染个性化配置页面
@viewer.route('/config', methods=['GET'])
def config():
    #from applications.application import application
    application = get_application()
    individuation = get_individuation()
    # 获取手势名称
    gesture_config = individuation.get_gesture_names()
    # print("gesture_names:", gesture_names)
    print("gesture_config:", gesture_config)
    # 获取应用功能名称
    application_names = application.get_application_names()
    # print("application_names:", application_names)
    # 根据应用程序名称设置 text_list
    #text_list = {application_names[i]: ["你好"] for i in range(len(application_names))}
    text_list = individuation.get_speech_individuation_dict()
    print("text_list:", text_list)
    # 根据应用功能和手势名称设置 gesture_data
    #gesture_data = {gesture_names[i]: application_names for i in range(len(gesture_names))}
    #print("gesture_data:", gesture_data)
    # 返回页面并渲染配置
    return render_template('config.html', text_list=text_list, gesture_data=gesture_config)

# 渲染权限设置页面
@viewer.route('/settings')
def settings():
    print(" 已跳转到权限设置页面")
    try:
        setting = get_setting()
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

def generate_frames():
    global camera_mgr
    if camera_mgr is None:
        camera_mgr = get_camera_manager()
    result = camera_mgr.initialize_camera(0, 640, 480, False)
    
    #time.sleep(1.0)

    
    if not camera_mgr.is_running():
        print("Could not start camera. Returning error frame.")
        error_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_img, "Camera Error", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 0, 255), 2, cv2.LINE_AA)
        ret, buffer = cv2.imencode('.jpg', error_img)
        if ret:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            return

    print("Starting video feed stream")
    frame_count = 0
    error_count = 0
    max_errors = 10

    while True:
        try:
            success, frame = camera_mgr.read_frame()
            if not success:
                error_count += 1
                if error_count > max_errors:
                    print(f"Too many errors ({error_count}) reading frames. Restarting camera...")
                    camera_mgr.release_camera()

                    #time.sleep(1.0)
  
                    camera_mgr.initialize_camera(0, 640, 480, False)
                    error_count = 0
                    continue

                error_img = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(error_img, f"Frame Error {error_count}", (50, 240), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                ret, buffer = cv2.imencode('.jpg', error_img)
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

                #time.sleep(0.5)


                continue
            
            error_count = 0
            frame_count += 1
        
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
                
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


            #time.sleep(0.01)
            
        except Exception as e:
            print(f"Error in generate_frames: {str(e)}")
            error_count += 1
            if error_count > max_errors:
                print("Too many errors. Stopping video feed.")
                break

            #time.sleep(0.5)


@viewer.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@viewer.route('/image')
def get_image():
    camera_mgr = get_camera_manager()
    result = camera_mgr.initialize_camera(0, 640, 480, False)

    time.sleep(0.5)
    
    if not camera_mgr.is_running():
        error_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_img, "Camera Not Available", (50, 240), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        ret, buffer = cv2.imencode('.jpg', error_img)
        if ret:
            return Response(buffer.tobytes(), mimetype='image/jpeg')
        return jsonify({'error': 'Camera not available'}), 500
    
    success, frame = camera_mgr.read_frame()
    if not success:
        error_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_img, "Failed to Capture Frame", (50, 240), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        ret, buffer = cv2.imencode('.jpg', error_img)
        if ret:
            return Response(buffer.tobytes(), mimetype='image/jpeg')
        return jsonify({'error': 'Failed to capture frame'}), 500
    
    ret, buffer = cv2.imencode('.jpg', frame)
    if not ret:
        return jsonify({'error': 'Failed to encode image'}), 500
    
    return Response(buffer.tobytes(), mimetype='image/jpeg')

# 控制提示灯状态
light_color = "green"
light_blink = False

# 控制提示灯状态(对外调用)
def update_light(color, blink):
    print(f"更新提示灯颜色: {color}, 闪烁状态: {blink}")
    global light_color
    global light_blink
    light_color = color
    light_blink = blink

@viewer.route('/get_light', methods=['GET'])
def get_light():
    global light_color
    global light_blink
    return jsonify({'color': light_color, 'blink': light_blink})

# 在提示框输出文字
latest_message = "车载多模态智能交互系统初始化完毕!"

# 在提示框输出文字(对外调用)
def update_note(note):
    print(f"更新提示框内容: {note}")
    global latest_message
    latest_message = note

@viewer.route('/get_note', methods=['GET'])
def get_note():
    return jsonify({'updated_message': latest_message})

# 播放音乐
@viewer.route('/play_music', methods=['POST'])
def play_music():
    from applications.application import application
    data = request.get_json()
    music_name = data.get('music')
    print(f"播放音乐：{music_name}")
    application.schedule(application.type.music_play, [music_name])
    return '', 204  # No Content

# 暂停/继续播放音乐
@viewer.route('/pause_music', methods=['POST'])
def pause_music():
    from applications.application import application
    print("暂停或继续播放音乐")
    application.schedule(application.type.music_change_pause, [])
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
    individuation = get_individuation()
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

# 外部控制页面跳转
last_action = None

def jump_to_page(page_name):
    if page_name in ['music', 'navigation', 'status']:
        print(f"外部请求跳转到: {page_name}")
        global last_action
        last_action = page_name
        # requests.post('http://127.0.0.1:5000/trigger_action', json={'action': page_name})
    else:
        print("非法的外部跳转请求")

@viewer.route('/trigger_action', methods=['POST'])
def trigger_action():
    data = request.get_json()
    action = data.get('action')
    if action in ['music', 'navigation', 'status']:
        print(f"收到 POST 请求：{action} {url_for(action)}")
        # return redirect(url_for(action))  # 自动跳转到对应的页面
        return jsonify({'status': 'ok', 'action': action}) , 200
    else:
        return jsonify({'status': 'error', 'message': 'Unknown action'}), 400

@viewer.route('/get_action')
def get_action():
    global last_action
    action = last_action
    last_action = None  # 用后清除
    return jsonify({'action': action})

# 外部控制导航
# 自动导航参数
#auto_navigation_params = None
navigation_from = None
navigation_to = None
# 添加供外部调用的自动导航功能
def navigate(from_ = "南开大学津南校区(地铁站)", to_ = "南开大学八里台(地铁站)"):
    # 设置导航参数
    global navigation_from, navigation_to
    navigation_from = from_
    navigation_to = to_
    # 先跳转到导航页面
    # jump_to_page('navigation')
    time.sleep(1)

    return True
    
@viewer.route('/get_navigation', methods=['GET'])
def get_navigation():
    """获取自动导航参数，前端页面加载时调用"""
    global navigation_from, navigation_to
    params = {
        'start': navigation_from,
        'end': navigation_to
    }
    navigation_from = None
    navigation_to = None
    return jsonify({'params': params})

# 修改init_viewer函数
def init_viewer():
    print("[已废弃]此函数已不再使用")
    pass

# 添加新的服务器启动函数
def start_flask_server():
    print("[重要]启动Flask服务器", threading.get_ident())
    global camera_mgr
    if camera_mgr is None:
        camera_mgr = get_camera_manager()
    viewer.run(debug=False)

@viewer.route('/call_set_user', methods=['POST'])
def set_user():
    data = request.get_json()
    username = data.get('status', '空')
    print(f"设置用户,用户名为\"{username}\"")
    setting = get_setting()
    setting.register_voiceprint(username)
    return '', 204

@viewer.route('/call_delete_user', methods=['POST'])
def delete_user():
    data = request.get_json()
    username = data.get('username', '空')
    print(f"删除用户,用户名为\"{username}\"")
    setting = get_setting()
    setting.delete_voiceprint(username)
    return '', 204

@viewer.route('/call_set_driver', methods=['POST'])
def set_driver():
    data = request.get_json()
    driver_name = data.get('drivername', None)
    print(f"设置驾驶员,用户名为\"{driver_name}\"")
    setting = get_setting()
    setting.set_driver(driver_name)
    return '', 204

def my_function(name):
    print(f"线程 {name} 正在运行")
    while True:
        time.sleep(10)  # 模拟耗时操作
        update_light("red", True)
        time.sleep(10)  # 模拟耗时操作
        update_light("red", False)
        time.sleep(10)  # 模拟耗时操作
        update_light("green", True)
        time.sleep(10)  # 模拟耗时操作
        update_light("green", False)
    print(f"线程 {name} 已完成")

if __name__ == '__main__':
    #import threading
    #thread = threading.Thread(target=my_function, args=("示例线程",))
    # 启动线程
    #thread.start()
    viewer.run(debug=True)
