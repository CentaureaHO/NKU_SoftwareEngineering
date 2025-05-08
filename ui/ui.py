from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/music')
def music():
    return render_template('music.html')

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
