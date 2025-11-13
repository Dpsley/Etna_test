import pandas as pd
import requests
from flask import Flask, request, jsonify
import os

from forecasters.main import new_forecast
from pipelines.main import load_pipline_from_dump
from trainers.main import fitter
import json
from dotenv import load_dotenv

load_dotenv()

DATA_CSV = os.getenv("MAIN_CSV_SRC")

app = Flask(__name__)


@app.route('/train_refresh', methods=['POST'])
def train_refresh():
    try:
        fitter()
        return jsonify({"status": True, "error": None}), 200
    except Exception as e:
        return jsonify({"status": False, "error": str(e)}), 500

@app.route('/upload_dataset', methods=['POST'])
def upload_dataset():
    if 'file' not in request.files:
        return jsonify({"status": False, "error": "Файл не найден в запросе"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": False, "error": "Имя файла пустое"}), 400

    try:
        file.save(DATA_CSV)  # сохраняем прямо в DATA_CSV
        df = pd.read_csv(DATA_CSV, sep="|", encoding="utf-8")
        df = df.dropna(axis=1, how='all')

        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        df['segment'] = df['department'] + '|' + df['article']
        df["productProperties"] = df["productProperties"].apply(json.loads)

        # разваливаем словари в колонки
        props_df = pd.json_normalize(df["productProperties"])
        props_df = props_df.add_prefix("prop_")  # чтобы не путалось

        if "target" in df.columns:
            df["target"] = df["target"].apply(lambda x: max(0, float(x)))

        df.drop(["article", "productProperties"], axis=1,
                inplace=True)

        df = pd.concat([df, props_df], axis=1)

        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].replace('', 'unknown')  # вот это главное
                df[col] = df[col].fillna("unknown")
            else:
                df[col] = df[col].fillna(0)

        df.to_csv(DATA_CSV, sep=",", index=False)
        return jsonify({"status": True, "error": None}), 200
    except Exception as e:
        return jsonify({"status": False, "error": str(e)}), 500

@app.route('/forecast', methods=['POST'])
def forecast():
    pipeline= load_pipline_from_dump()
    new_forecast(pipeline)
    with open("forecast.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


if __name__ == "__main__":
    # Выполняем миграцию
    port = int(os.environ.get("PORT", 7777))
    app.run(debug=False, port=port, host="0.0.0.0")
