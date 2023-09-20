from flask import Flask, render_template, request, make_response, g
import os
import socket
import random
import json
import logging

option_a = os.getenv("OPTION_A", "Cats")
option_b = os.getenv("OPTION_B", "Dogs")
hostname = socket.gethostname()

app = Flask(__name__)

#gunicorn_error_logger = logging.getLogger("gunicorn.error")
#app.logger.handlers.extend(gunicorn_error_logger.handlers)
#app.logger.setLevel(logging.INFO)

@app.route("/success/<name>")
def success(name):
    return "welcome %s" % name

@app.route("/login", methods=["POST", "GET"])
def login():
    pass

@app.route("/", methods=["POST","GET"])
def hello():
    return "hello"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True, threaded=True)