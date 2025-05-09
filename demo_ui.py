import requests
import json

def update_config(new_text_list, new_gesture_data):
    url = "http://127.0.0.1:5000/update_config"  # Flask 后端的地址
    
    # 构建请求的 payload（请求体）
    payload = {
        "text_list": new_text_list,
        "gesture_data": new_gesture_data
    }
    
    # 发送 POST 请求
    try:
        response = requests.post(url, json=payload)
        
        # 检查请求是否成功
        if response.status_code == 200:
            print("配置已更新成功！")
            print("响应内容：", response.json())
        else:
            print("更新失败，状态码：", response.status_code)
            print("响应内容：", response.text)
    except requests.exceptions.RequestException as e:
        print("请求发送失败:", e)

# 示例数据
new_text_list = {
    '语音识别': ["启用", "禁用", "自动"],
    '语音控制': ["开启", "关闭"]
}

new_gesture_data = {
    "1111": ["选项AAAA", "选项2"],
    "右转": ["选项X", "选项Y"],
    "停止": ["选项I", "选项II"]
}

# 调用函数更新配置
update_config(new_text_list, new_gesture_data)
