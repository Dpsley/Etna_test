from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from datetime import datetime, timedelta

app = Flask(__name__)

MODEL_PATH = "catboost_model.cbm"
DATA_CSV = "expanded.csv"

# Загружаем модель один раз
model = CatBoostRegressor()
model.load_model(MODEL_PATH)

# Категориальные фичи
CAT_FEATURES = [
    'Department', 'Article', 'ProductName', 'ProductGroup', 'ProductType',
    'Вид', 'Регион', 'Тип', 'Марка', 'Код производителя', 'Вкус', 'Ароматизированный'
]

# Генерация лагов и скользящих средних
def generate_features(df, max_lag=28):
    df = df.sort_values('Date')
    for lag in [1, 7, 14, 28]:
        df[f'sold_lag{lag}'] = df['Sold'].shift(lag)
    for ma in [7, 14]:
        df[f'sold_ma{ma}'] = df['Sold'].rolling(ma).mean().shift(1)
    return df

# Обработка категориальных фичей
def process_cat_features(df):
    for col in CAT_FEATURES:
        df[col] = df[col].fillna('nan').astype(str)
    return df

@app.route("/forecast", methods=["POST"])
def forecast():
    try:
        data = request.json
        start_date = datetime.fromisoformat(data['start_date'])
        end_date = datetime.fromisoformat(data['end_date'])

        df = pd.read_csv(DATA_CSV, sep=';', parse_dates=['Date'], dayfirst=True)

        # Генерим признаки
        df = generate_features(df)
        df = process_cat_features(df)

        # Фильтруем по дате
        last_date = df['Date'].max()
        future_dates = pd.date_range(start=last_date + timedelta(days=1),
                                     end=end_date)
        forecast_rows = []

        for d in future_dates:
            # Для каждого дня создаем строку на основе последнего состояния
            last_row = df[df['Date'] == last_date].copy()
            last_row['Date'] = d

            # обновляем лаги
            for lag in [1, 7, 14, 28]:
                last_row[f'sold_lag{lag}'] = df.loc[df['Date'] == last_date - timedelta(days=lag), 'Sold'].values[0] \
                    if not df.loc[df['Date'] == last_date - timedelta(days=lag), 'Sold'].empty else 0
            for ma in [7, 14]:
                last_row[f'sold_ma{ma}'] = df.loc[df['Date'] <= last_date].tail(ma)['Sold'].mean()

            forecast_rows.append(last_row)
            last_date = d

        df_forecast = pd.concat(forecast_rows, ignore_index=True)
        df_forecast = process_cat_features(df_forecast)

        # Предсказание
        X_pred = df_forecast.drop(columns=['Sold'])
        y_pred = model.predict(X_pred)

        df_forecast['PredictedSold'] = y_pred

        # Возвращаем JSON
        result = df_forecast[['Date', 'PredictedSold']].to_dict(orient='records')
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
