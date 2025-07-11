#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# _author = 'Fangyi Liu','Yidian Lin','Xiaoyu Guo'

"""
Module Description:
    用于实现车载多模态智能交互系统的前端功能
"""

import os
import sys
import time
from dataclasses import dataclass

import cv2
import numpy as np
import requests
from flask import Flask, Response, jsonify, render_template, request, url_for

from utils.camera_manager import get_camera_manager
from components import get_component # Moved from start_flask_server

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

viewer = Flask(__name__)

@dataclass
class AppState:
    """封装应用的可变状态"""
    def __init__(self):
        self.camera_mgr = None
        self.application = None
        self.light_color = "green"
        self.light_blink = False
        self.latest_message = "车载多模态智能交互系统初始化完毕!"
        self.last_action = None
        self.navigation = None

app_state = AppState()

def get_application():
    """获取应用程序控制器"""
    return get_component("application")


def get_controller():
    """获取多模态控制器"""
    return get_component("controller")

def get_setting():
    """获取设置模块"""
    return get_component("setting")


def get_individuation():
    """获取个性化模块"""
    return get_component("individuation")


# Amap Web服务API密钥
AMAP_WEB_SERVICE_KEY = "3479b328113102a27c20e818c3bc143c"

# Amap Web服务API的基本URL
GEOCODE_API_URL = "https://restapi.amap.com/v3/geocode/geo"
DRIVING_API_URL = "https://restapi.amap.com/v3/direction/driving"


def get_coords(address):
    """使用Amap Geocoding API从地址获取坐标的辅助函数"""
    params = {"address": address, "key": AMAP_WEB_SERVICE_KEY}
    try:
        response = requests.get(GEOCODE_API_URL, params=params, timeout=5)
        response.raise_for_status()  # Raise an exception for bad status codes
        result = response.json()
        if result and result["status"] == "1" and result["geocodes"]:
            # Return the location string (longitude,latitude)
            return result["geocodes"][0]["location"]

        print(
            f"Geocoding failed for {address}: {
                result.get('info', 'Unknown error')}"
        )
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error calling Geocoding API for {address}: {e}")
        return None


@viewer.route("/")
def index():
    """渲染索引页面"""
    return render_template("index.html")


@viewer.route("/music")
def music():
    """渲染音乐页面"""
    print("已跳转到音乐页面")
    try:
        music_info = app_state.application.schedule(app_state.application.type.music_getlist, [])
    except ImportError as e:
        print(f"获取音乐列表失败: {e}")
        music_info = []
    return render_template("music.html", music_info=music_info)


@viewer.route("/navigation")
def navigation():
    """渲染导航页面"""
    print("已跳转到导航页面")
    return render_template("navigation.html")


@viewer.route("/status")
def status():
    """渲染车辆状态监测页面"""
    status_info = app_state.application.schedule(app_state.application.type.monitor_getlist, [])
    oil_quantity = status_info[0]
    tire_pressure = status_info[1]
    mileage = status_info[2]
    # print("status_info:", status_info)
    return render_template(
        "status.html",
        oil_quantity=oil_quantity,
        tire_pressure=tire_pressure,
        mileage=mileage,
    )


@viewer.route("/config", methods=["GET"])
def config():
    """渲染个性化配置页面"""
    individuation = get_individuation()
    # 获取手势配置信息
    gesture_config = individuation.get_gesture_names()
    print("gesture_config:", gesture_config)
    # 获取语音输入框的内容
    text_list = individuation.get_speech_individuation_dict()
    print("text_list:", text_list)
    # 返回页面并渲染配置
    return render_template(
        "config.html", text_list=text_list, gesture_data=gesture_config
    )


@viewer.route("/settings")
def settings():
    """渲染权限设置页面"""
    print(" 已跳转到权限设置页面")
    try:
        setting = get_setting()
        music_info = setting.get_voiceprints()
        driver_info = setting.get_driver()
    except ImportError as e:
        print(f"获取声纹列表/驾驶员失败: {e}")
        music_info = []
        driver_info = None

    if not driver_info:
        driver_info = "无"
    print("music_info:", music_info)
    print("driver_info:", driver_info)
    return render_template(
        "settings.html", music_info=music_info, driver_info=driver_info
    )


def generate_frames():
    """获取前置摄像头"""
    app_state.camera_mgr.initialize_camera(0, 640, 480, False)

    if not app_state.camera_mgr.is_running():
        print("Could not start camera. Returning error frame.")
        error_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(
            error_img,
            "Camera Error",
            (50, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        ret, buffer = cv2.imencode(".jpg", error_img)
        if ret:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )
            return

    print("Starting video feed stream")
    frame_count = 0
    error_count = 0
    max_errors = 10

    while True:
        try:
            success, frame = app_state.camera_mgr.read_frame()
            if not success:
                error_count += 1
                if error_count > max_errors:
                    print(
                        f"Too many errors ({error_count}) reading frames. Restarting camera..."
                    )
                    app_state.camera_mgr.release_camera()

                    # time.sleep(1.0)

                    app_state.camera_mgr.initialize_camera(0, 640, 480, False)
                    error_count = 0
                    continue

                error_img = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(
                    error_img,
                    f"Frame Error {error_count}",
                    (50, 240),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                ret, buffer = cv2.imencode(".jpg", error_img)
                if ret:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
                    )

                # time.sleep(0.5)

                continue

            error_count = 0
            frame_count += 1

            ret, buffer = cv2.imencode(".jpg", frame)
            if not ret:
                continue

            frame_bytes = buffer.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )

            # time.sleep(0.01)

        except Exception as e: # pylint: disable=broad-exception-caught
            print(f"Error in generate_frames: {str(e)}")
            error_count += 1
            if error_count > max_errors:
                print("Too many errors. Stopping video feed.")
                break

            # time.sleep(0.5)


@viewer.route("/video_feed")
def video_feed():
    """获取前置摄像头"""
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


def update_light(color, blink):
    """更新提示灯状态(供外部调用的接口)"""
    print(f"更新提示灯颜色: {color}, 闪烁状态: {blink}")
    # global LIGHT_COLOR # pylint: disable=global-statement (移除)
    # global LIGHT_BLINK # pylint: disable=global-statement (移除)
    app_state.light_color = color
    app_state.light_blink = blink


@viewer.route("/get_light", methods=["GET"])
def get_light():
    """获取提示灯状态"""
    return jsonify({"color": app_state.light_color, "blink": app_state.light_blink})


def update_note(note):
    """更新提示框内容(供外部调用的接口)"""
    print(f"更新提示框内容: {note}")
    # global LATEST_MESSAGE # pylint: disable=global-statement (移除)
    app_state.latest_message = note


@viewer.route("/get_note", methods=["GET"])
def get_note():
    """更新提示框内容"""
    return jsonify({"updated_message": app_state.latest_message})


@viewer.route("/play_music", methods=["POST"])
def play_music():
    """播放音乐"""
    data = request.get_json()
    music_name = data.get("music")
    print(f"播放音乐：{music_name}")
    app_state.application.schedule(app_state.application.type.music_play, [music_name])
    return "", 204  # No Content


@viewer.route("/pause_music", methods=["POST"])
def pause_music():
    """暂停或继续播放音乐"""
    print("暂停或继续播放音乐")
    app_state.application.schedule(app_state.application.type.music_change_pause, [])
    return "", 204


@viewer.route("/call_navigate", methods=["POST"])
def call_navigate():
    """进行导航"""
    data = request.get_json()
    start_location = data.get("start")
    end_location = data.get("end")

    if not start_location or not end_location:
        return (
            jsonify({"error": "Please provide start and end locations"}),
            400,
        )  # Bad Request

    # Get coordinates for start and end locations
    start_coords = get_coords(start_location)
    end_coords = get_coords(end_location)

    if not start_coords:
        return (
            jsonify(
                {"error": f"Could not geocode start location: {start_location}"}),
            404,
        )
    if not end_coords:
        return (
            jsonify({"error": f"Could not geocode end location: {end_location}"}),
            404,
        )

    # Call Amap Driving Route Planning API
    driving_params = {
        "origin": start_coords,
        "destination": end_coords,
        "key": AMAP_WEB_SERVICE_KEY,
        "extensions": "all",  # Request detailed route information
    }

    try:
        driving_response = requests.get(
            DRIVING_API_URL, params=driving_params, timeout=5)
        driving_response.raise_for_status()
        driving_result = driving_response.json()

        if (
            driving_result
            and driving_result["status"] == "1"
            and driving_result["route"]
        ):
            # Return the route data to the frontend
            return jsonify(driving_result)
        error_info = driving_result.get("info", "Unknown route planning error")
        print(f"Driving route planning failed: {error_info}")
        return (
            jsonify({"error": f"Driving route planning failed: {error_info}"}),
            500,
        )  # Internal Server Error

    except requests.exceptions.RequestException as e:
        print(f"Error calling Driving API: {e}")
        return jsonify({"error": f"Error calling Driving API: {e}"}), 500


# 外部控制页面跳转


def jump_to_page(page_name):
    """页面跳转(供外部调用的接口)"""
    if page_name in ["music", "navigation", "status"]:
        print(f"外部请求跳转到: {page_name}")
        # global LAST_ACTION # pylint: disable=global-statement (移除)
        app_state.last_action = page_name
    else:
        print("非法的外部跳转请求")


@viewer.route("/trigger_action", methods=["POST"])
def trigger_action():
    """处理外部请求的跳转动作"""
    data = request.get_json()
    action = data.get("action")
    if action in ["music", "navigation", "status"]:
        print(f"收到 POST 请求：{action} {url_for(action)}")
        return jsonify({"status": "ok", "action": action}), 200

    return jsonify({"status": "error", "message": "Unknown action"}), 400


@viewer.route("/get_action")
def get_action():
    """获取的跳转动作"""
    # global LAST_ACTION # pylint: disable=global-statement (移除)
    action = app_state.last_action
    app_state.last_action = None
    return jsonify({"action": action})


# 导航参数


def navigate(from_="南开大学津南校区(地铁站)", to_="南开大学八里台(地铁站)"):
    """自动导航功能(供外部调用的接口)"""
    # 设置导航参数
    # global NAVIGATION_FROM, NAVIGATION_TO # pylint: disable=global-statement (移除)
    app_state.navigation = (from_, to_)
    time.sleep(1)
    return True


@viewer.route("/get_navigation", methods=["GET"])
def get_navigation():
    """获取自动导航参数，前端页面加载时调用"""
    # global NAVIGATION_FROM, NAVIGATION_TO # pylint: disable=global-statement (移除)
    params = {"start": app_state.navigation[0],
              "end": app_state.navigation[1]} if app_state.navigation else {}
    app_state.navigation = None
    return jsonify({"params": params})


@viewer.route("/save_config", methods=["POST"])
def save_config():
    """保存配置信息"""
    data = request.get_json()
    individuation = get_individuation()
    # 获取语音输入框的内容
    voice_inputs = data.get("voiceInputs", {})
    print("保存的语音功能配置:")
    for key, value in voice_inputs.items():
        print(f"{key}: {value}")
    individuation.set_speech_individuation(voice_inputs)

    # 获取手势功能的选择
    gesture_options = data.get("gestureOptions", {})
    print("保存的手势功能配置:")
    for key, value in gesture_options.items():
        print(f"{key}: {value}")
    individuation.set_gesture_individuation(gesture_options)

    return jsonify({"status": "ok", "message": "配置已保存"})


@viewer.route("/call_set_user", methods=["POST"])
def set_user():
    """设置用户"""
    data = request.get_json()
    username = data.get("status", "空")
    print(f'设置用户,用户名为"{username}"')
    setting = get_setting()
    setting.register_voiceprint(username)
    return "", 204


@viewer.route("/call_delete_user", methods=["POST"])
def delete_user():
    """删除用户"""
    data = request.get_json()
    username = data.get("username", "空")
    print(f'删除用户,用户名为"{username}"')
    setting = get_setting()
    setting.delete_voiceprint(username)
    return "", 204


@viewer.route("/call_set_driver", methods=["POST"])
def set_driver():
    """设置驾驶员"""
    data = request.get_json()
    driver_name = data.get("drivername", None)
    print(f'设置驾驶员,用户名为"{driver_name}"')
    setting = get_setting()
    setting.set_driver(driver_name)
    return "", 204


def start_flask_server():
    """启动Flask服务器"""
    print("启动Flask服务器")
    # global CAMERA_MGR # pylint: disable=global-statement (移除)
    # global APPLICATION # pylint: disable=global-statement (移除)
    if app_state.camera_mgr is None:
        app_state.camera_mgr = get_camera_manager()
    # from components import get_component # Moved to top
    if app_state.application is None:
        app_state.application = get_component("application")
    viewer.run(debug=False)
