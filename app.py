import requests
from flask import Flask, request, jsonify
import os

from forecasters.main import new_forecast
from pipelines.main import load_pipline_from_dump
from trainers.main import fitter

app = Flask(__name__)


@app.route('/train_refresh', methods=['POST'])
def train_refresh():
    fitter()
    return True

@app.route('/forecast', methods=['POST'])
def forecast():
    pipeline= load_pipline_from_dump()
    new_forecast(pipeline)
    return True


if __name__ == "__main__":
    # Выполняем миграцию
    port = int(os.environ.get("PORT", 7777))
    app.run(debug=False, port=port, host="0.0.0.0")
