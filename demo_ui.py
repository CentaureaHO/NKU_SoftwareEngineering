import time
import requests
import threading
import webbrowser
from viewer.viewer import init_viewer

if __name__ == '__main__':
    viewer_thread = threading.Thread(target=init_viewer)
    viewer_thread.daemon = True  # 设置为守护线程，主线程退出时自动结束
    viewer_thread.start()
    webbrowser.open("http://127.0.0.1:5000")
    
    count = 0
    while True:
        count += 1
        time.sleep(5)
        # 关闭闪烁（变为绿色常亮）
        requests.post('http://127.0.0.1:5000/set_blinking', json={'enabled': False})
        print("🔵 已关闭闪烁")
        time.sleep(5)  # 继续观察状态

        # 开启闪烁（变为红色闪烁）
        requests.post('http://127.0.0.1:5000/set_blinking', json={'enabled': True})
        print("🔴 已开启闪烁")
        time.sleep(5)  # 继续观察状态

        response = requests.post('http://127.0.0.1:5000/update_string', json={'message': str(count)})
        # 输出返回的消息
        if response.status_code == 200:
            print('服务器返回的消息:', response.json()['updated_message'])
        else:
            print('请求失败，状态码:', response.status_code)
