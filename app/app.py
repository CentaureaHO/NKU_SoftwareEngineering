from flask import Flask, render_template  # 加上 render_template

# 保持你已有的蓝图导入
from routes.voice import voice_bp
from routes.vision import vision_bp
from routes.fusion import fusion_bp
from routes.control import control_bp

app = Flask(__name__)

# 注册蓝图
app.register_blueprint(voice_bp, url_prefix='/voice')
app.register_blueprint(vision_bp, url_prefix='/vision')
app.register_blueprint(fusion_bp, url_prefix='/fusion')
app.register_blueprint(control_bp, url_prefix='/control')

# 修改主页返回 HTML 模板
@app.route('/')
def index():
    return render_template('index.html')  # 渲染 templates/index.html

if __name__ == '__main__':
    app.run(debug=True)
