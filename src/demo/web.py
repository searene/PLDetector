from flask import Flask, render_template, request

import sys
sys.path.insert(0, '../..')

from src.detector import detect

app = Flask(__name__, template_folder="./template")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/detect")
def detect_code():
    code = request.args.get("code")
    return detect(code)


if __name__ == "__main__":
    app.run(port=5001, debug=True)
