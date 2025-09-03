from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import json

app = Flask(__name__)

# Загрузка модели
model = CatBoostRegressor()
model.load_model('catboost_model.cbm')

# Функция для генерации признаков
def generate_features(df, lag_days, ma_windows):
    # Лаги
    for lag in lag_days:
        df[f'sold_lag{lag}'] = df['Sold'].shift(lag).fillna(0)

    # Скользящие средние
    for w in ma_windows:
        df[f'sold_ma{w}'] = df['Sold'].rolling(w).mean().fillna(0)
        df[f'sold_median{w}'] = df['Sold'].rolling(w).median().fillna(0)

    # Волатильность
    df["sold_std7"] = df["Sold"].rolling(7).std().fillna(0)
    df["sold_std30"] = df["Sold"].rolling(30).std().fillna(0)

    # Бинарные признаки
    df['sold_last_day_binary'] = (df['sold_lag1'] > 0).astype(int)
    df['sold_last_week_binary'] = (df['sold_lag7'] > 0).astype(int)
    df['sold_last_month_binary'] = (df['sold_lag28'] > 0).astype(int)

    # Преобразование категориальных признаков
    df['brand_region'] = df['Марка'].astype(str) + "_" + df['Регион'].astype(str)
    df['type_flavor'] = df['Тип'].astype(str) + "_" + df['Вкус'].astype(str)

    df['quarter'] = df['Date'].dt.quarter
    df['is_start_of_month'] = (df['Date'].dt.day <= 3).astype(int)
    df['is_end_of_month'] = (df['Date'].dt.day >= 27).astype(int)

    return df

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Получение данных из запроса
        data = request.get_json()
        article = data['article']
        start_date = pd.to_datetime(data['start_date'])
        end_date = pd.to_datetime(data['end_date'])

        # Фильтрация данных по артикулу
        df = pd.read_csv('expanded.csv', sep=';', parse_dates=['Date'], dayfirst=False)
        df = df[df['Article'] == article]

        # Генерация признаков
        lag_days = [1, 2, 3, 7, 14, 21, 28, 60, 90]
        ma_windows = [3, 7, 14, 21, 28, 60, 90]
        df = generate_features(df, lag_days, ma_windows)

        # Фильтрация данных по диапазону дат
        df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

        # Подготовка данных для предсказания
        features = [
            "ProductGroup", "ProductType", "Вид", "Регион", "Тип",
            "Марка", "Код производителя", "Вкус", "Ароматизированный",
            "brand_region", "type_flavor", "Department",
            *[f'sold_lag{l}' for l in lag_days],
            *[f'sold_ma{w}' for w in ma_windows],
            *[f'sold_median{w}' for w in ma_windows],
            "sold_std7", "sold_std30", "stock_diff", "restock_flag",
            "days_since_last_restock", "Reserve", "Available",
            "sold_last_day_binary", "sold_last_week_binary",
            "sold_last_month_binary", "day_of_week", "is_weekend",
            "month", "day", "quarter", "is_start_of_month", "is_end_of_month"
        ]
        X = df[features]

        # Преобразование категориальных признаков
        cat_features = ["Department"] + ["ProductGroup", "ProductType", "Вид", "Регион", "Тип",
                                         "Марка", "Код производителя", "Вкус", "Ароматизированный",
                                         "brand_region", "type_flavor"]
        for col in cat_features:
            X[col] = X[col].fillna("NA").astype(str)

        # Предсказание
        y_pred_log = model.predict(X)
        y_pred = np.abs(np.round(np.expm1(y_pred_log)))

        # Формирование ответа
        response = {
            "article": article,
            "predictions": [int(x) for x in y_pred],
            "sum_predictions": int(y_pred.sum()),
            "dates": df['Date'].dt.strftime('%Y-%m-%d').tolist()
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
