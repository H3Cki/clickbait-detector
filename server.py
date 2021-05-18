from flask import Flask
app = Flask(__name__)


@app.route('/predict/<text>')
def hello_name(text):
    return f"PREDICTED: {text}"

if __name__ == '__main__':
    app.debug = True
    app.run(host='192.168.1.2', ssl_context='adhoc')