import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
import joblib
from catboost import CatBoostRegressor

app = Flask(__name__)

# Загрузка модели
model = CatBoostRegressor()
model.load_model("catboost_model.cbm")
# Загружаем исходную историю
history_df = pd.read_csv("expanded.csv", sep=";", parse_dates=['Date'])

# Список фичей для модели
features = [
    "sold_lag1", "sold_lag2", "sold_lag3", "sold_ma3", "sold_median7", "sold_std30",
    "day_of_week", "is_weekend", "month", "day", "quarter", "is_start_of_month",
    "is_end_of_month", "brand_region", "type_flavor"
]

def compute_features(g):
    # лаги
    sold = g["Sold"].tolist()
    return pd.DataFrame({
        "sold_lag1": [sold[-1] if len(sold) >= 1 else 0],
        "sold_lag2": [sold[-2] if len(sold) >= 2 else 0],
        "sold_lag3": [sold[-3] if len(sold) >= 3 else 0],
        "sold_ma3": [np.mean(sold[-3:]) if len(sold) >= 1 else 0],
        "sold_median7": [np.median(sold[-7:]) if len(sold) >= 1 else 0],
        "sold_std30": [np.std(sold[-30:]) if len(sold) >= 1 else 0]
    })

@app.route("/predict", methods=["POST"])
def predict():
    global history_df

    data = request.json
    start_date = pd.to_datetime(data["start_date"])
    end_date = pd.to_datetime(data["end_date"])

    results = []

    for date in pd.date_range(start_date, end_date):
        df_date = history_df.groupby(['Department','Article'], group_keys=False).apply(
            lambda g: pd.DataFrame({
                "Department": g["Department"].iloc[-1],
                "Article": g["Article"].iloc[-1],
                "ProductName": g["ProductName"].iloc[-1],
                "ProductGroup": g["ProductGroup"].iloc[-1],
                "ProductType": g["ProductType"].iloc[-1],
                "Вид": g["Вид"].iloc[-1],
                "Регион": g["Регион"].iloc[-1],
                "Тип": g["Тип"].iloc[-1],
                "Марка": g["Марка"].iloc[-1],
                "Код производителя": g["Код производителя"].iloc[-1],
                "Вкус": g["Вкус"].iloc[-1],
                "Ароматизированный": g["Ароматизированный"].iloc[-1],
            }, index=[0])
        ).reset_index(drop=True)

        # добавляем лаги и скользящие
        lag_features = history_df.groupby(['Department','Article'], group_keys=False).apply(compute_features).reset_index(drop=True)
        df_date = pd.concat([df_date.reset_index(drop=True), lag_features], axis=1)

        # календарные признаки
        df_date["day_of_week"] = date.dayofweek
        df_date["is_weekend"] = int(date.dayofweek >= 5)
        df_date["month"] = date.month
        df_date["day"] = date.day
        df_date["quarter"] = date.quarter
        df_date["is_start_of_month"] = int(date.day <= 3)
        df_date["is_end_of_month"] = int(date.day >= 27)

        # композитные признаки
        df_date["brand_region"] = df_date["Марка"].astype(str) + "_" + df_date["Регион"].astype(str)
        df_date["type_flavor"]  = df_date["Тип"].astype(str)   + "_" + df_date["Вкус"].astype(str)

        # предсказание
        y_pred_log = model.predict(df_date[features])
        y_pred = np.expm1(y_pred_log)

        # собираем результаты
        for i, row in df_date.iterrows():
            results.append({
                "Article": row["Article"],
                "ProductName": row["ProductName"],
                "Date": date.strftime("%Y-%m-%d"),
                "Predicted": float(y_pred[i])
            })

        # добавляем прогноз в историю для корректных лагов на следующую дату
        df_date_copy = df_date.copy()
        df_date_copy["Sold"] = y_pred.round().astype(int)
        df_date_copy["Date"] = date
        history_df = pd.concat([history_df, df_date_copy], ignore_index=True)

    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5555)
