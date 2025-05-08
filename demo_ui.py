import requests
import webbrowser

requests.post('http://127.0.0.1:5000/trigger_action', json={'action': 'music'})
#webbrowser.open("http://127.0.0.1:5000/music")
print(111)