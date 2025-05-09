# import requests

# url = 'http://127.0.0.1:5000/update_string'

# # 传入的字符串
# data = {
#     'message': '这是通过 requests 传入的字符串！'
# }

# # 发起 POST 请求
# response = requests.post(url, json=data)

# # 输出返回的消息
# if response.status_code == 200:
#     print('服务器返回的消息:', response.json()['updated_message'])
# else:
#     print('请求失败，状态码:', response.status_code)
import requests

requests.post('http://127.0.0.1:5000/update_string', json={'message': '🚨 外部发来的警告信息！'})
