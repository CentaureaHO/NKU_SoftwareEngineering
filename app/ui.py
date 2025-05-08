from flask import Flask, render_template
import sys
sys.path.append(r'C:/2025spring/软件工程/小组作业/NKU_SoftwareEngineering')  # 加入项目根目录
from applications.application import Application

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/music')
def music():
    # 生成的字符串，这里用示例字符串替代
    music_info = Application.schedule(Application.type.music_getlist,[])
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

if __name__ == '__main__':
    app.run(debug=True)
