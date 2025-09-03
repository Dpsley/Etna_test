from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor

app = Flask(__name__)

# Загружаем модель
model = CatBoostRegressor()
model.load_model("catboost_model.cbm")

# Те же списки фич, что и в обучении
product_features = [
    "ProductGroup","ProductType","Вид","Регион","Тип",
    "Марка","Код производителя","Вкус","Ароматизированный",
    "brand_region","type_flavor"
]
lag_days = [1,2,3,7,14,21,28,60,90]
ma_windows = [3,7,14,21,28,60,90]

features = product_features + [
    "Department",
    *[f'sold_lag{l}' for l in lag_days],
    *[f'sold_ma{w}' for w in ma_windows],
    *[f'sold_median{w}' for w in ma_windows],
    "sold_std7","sold_std30",
    "stock_diff","restock_flag","days_since_last_restock",
    "Reserve","Available",
    "sold_last_day_binary","sold_last_week_binary","sold_last_month_binary",
    "day_of_week","is_weekend","month","day",
    "quarter","is_start_of_month","is_end_of_month"
]
cat_features = ["Department"] + product_features

# Загружаем справочник с последними значениями по товарам
products_df = pd.read_csv("expanded.csv", sep=";")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    if "start_date" not in data or "end_date" not in data:
        return jsonify({"error": "start_date and end_date required"}), 400

    start_date = pd.to_datetime(data["start_date"])
    end_date = pd.to_datetime(data["end_date"])
    dates = pd.date_range(start=start_date, end=end_date)

    results = []
    for date in dates:
        df_date = products_df.copy()

        # Дата-фичи
        df_date["day_of_week"] = date.dayofweek
        df_date["is_weekend"] = int(date.dayofweek >= 5)
        df_date["month"] = date.month
        df_date["day"] = date.day
        df_date["quarter"] = date.quarter
        df_date["is_start_of_month"] = int(date.day <= 3)
        df_date["is_end_of_month"] = int(date.day >= 27)

        # Композитные фичи
        df_date["brand_region"] = df_date["Марка"].astype(str) + "_" + df_date["Регион"].astype(str)
        df_date["type_flavor"]  = df_date["Тип"].astype(str)   + "_" + df_date["Вкус"].astype(str)

        # Предсказание (лог → обратно в нормальные продажи)
        y_pred_log = model.predict(df_date[features])
        y_pred = np.expm1(y_pred_log)

        for i, row in df_date.iterrows():
            results.append({
                "Article": row["Article"],
                "ProductName": row["ProductName"],
                "Date": date.strftime("%Y-%m-%d"),
                "Predicted": float(y_pred[i])
            })

    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5555)
