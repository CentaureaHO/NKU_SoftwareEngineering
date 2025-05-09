import time
import requests
import threading
import webbrowser
from viewer.viewer import init_viewer

requests.post('http://127.0.0.1:5000/update_string', json={'message': '111111'})
