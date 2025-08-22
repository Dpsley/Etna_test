from flask import Flask, request, jsonify
import pandas as pd
from catboost import CatBoostRegressor
from datetime import datetime, timedelta

app = Flask(__name__)

# Загружаем модель
model = CatBoostRegressor()
model.load_model("catboost_model.cbm")

# Загружаем справочник товаров и последние данные для лагов/остатков
products_df = pd.read_csv("expanded_jan2024.csv", sep=";")
products_df = products_df.drop_duplicates(subset=["Article", "ProductName"], keep="last")

features = [
    "sold_lag1", "sold_lag7", "sold_ma7", "sold_ma14",
    "stock_diff", "restock_flag", "days_since_last_restock",
    "day_of_week", "is_weekend", "month"
]

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    if "start_date" not in data or "end_date" not in data:
        return jsonify({"error": "start_date and end_date fields are required"}), 400

    start_date = pd.to_datetime(data["start_date"])
    end_date = pd.to_datetime(data["end_date"])
    if start_date > end_date:
        return jsonify({"error": "start_date cannot be after end_date"}), 400

    # Формируем список всех дат в диапазоне
    dates = pd.date_range(start=start_date, end=end_date)
    results = []

    # Для каждого товара создаём словарь лагов/остатков
    lag_data = {}
    for _, row in products_df.iterrows():
        lag_data[row["Article"]] = {
            "ProductName": row["ProductName"],
            "sold_history": [row["sold_lag1"], row["sold_lag7"], row["sold_ma7"], row["sold_ma14"]],
            "stock_diff": row["stock_diff"],
            "restock_flag": row["restock_flag"],
            "days_since_last_restock": row["days_since_last_restock"]
        }

    for date in dates:
        for article, info in lag_data.items():
            sold_history = info["sold_history"]

            feat = {
                "sold_lag1": sold_history[0],
                "sold_lag7": sold_history[1],
                "sold_ma7": sold_history[2],
                "sold_ma14": sold_history[3],
                "stock_diff": info["stock_diff"],
                "restock_flag": info["restock_flag"],
                "days_since_last_restock": info["days_since_last_restock"],
                "day_of_week": date.dayofweek,
                "is_weekend": 1 if date.dayofweek >= 5 else 0,
                "month": date.month
            }

            pred = model.predict(pd.DataFrame([feat]))[0]
            results.append({
                "Article": article,
                "ProductName": info["ProductName"],
                "Date": date.strftime("%Y-%m-%d"),
                "Predicted": float(pred)
            })

            # Обновляем лаги для следующей даты
            sold_history[0] = pred
            sold_history[1] = sum(sold_history[-6:] + [pred]) / 7
            sold_history[2] = sum(sold_history[-6:] + [pred]) / 7
            sold_history[3] = sum(sold_history) / 4
            info["sold_history"] = sold_history

    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5555)
