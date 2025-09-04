import pandas as pd
import numpy as np
from datetime import timedelta
from catboost import CatBoostRegressor


def forecast(model, df, features, cat_features, horizon_date):
    horizon_date = pd.to_datetime(horizon_date)
    df_future = df.copy().reset_index(drop=True)
    last_date = df_future["Date"].max()

    lag_days = [1, 2, 3, 7, 14, 21, 28, 60, 90]
    ma_windows = [3, 7, 14, 21, 28, 60, 90]

    results = []

    while last_date < horizon_date:
        next_date = last_date + timedelta(days=1)

        # базовый ряд из последней строки
        row = df_future.iloc[-1].copy()
        row["Date"] = next_date
        row["Sold"] = 0  # временно

        # календарные фичи
        row["day_of_week"] = next_date.weekday()
        row["is_weekend"] = int(next_date.weekday() >= 5)
        row["month"] = next_date.month
        row["day"] = next_date.day
        row["quarter"] = next_date.quarter
        row["is_start_of_month"] = int(next_date.day <= 3)
        row["is_end_of_month"] = int(next_date.day >= 27)

        # лаги
        for lag in lag_days:
            if len(df_future) >= lag:
                row[f'sold_lag{lag}'] = df_future["Sold"].iloc[-lag]
            else:
                row[f'sold_lag{lag}'] = 0

        # скользящие
        for w in ma_windows:
            row[f'sold_ma{w}'] = df_future["Sold"].iloc[-w:].mean() if len(df_future) >= w else 0
            row[f'sold_median{w}'] = df_future["Sold"].iloc[-w:].median() if len(df_future) >= w else 0

        # бинарные признаки
        row['sold_last_day_binary']   = int(row['sold_lag1'] > 0)
        row['sold_last_week_binary']  = int(row['sold_lag7'] > 0)
        row['sold_last_month_binary'] = int(row['sold_lag28'] > 0)

        # категориальные
        for col in cat_features:
            row[col] = str(row[col]) if pd.notnull(row[col]) else "NA"

        # прогноз
        X_new = pd.DataFrame([row])[features]
        pred_log = model.predict(X_new)
        pred = np.expm1(pred_log).clip(min=0).round().astype(int)[0]

        row["Sold"] = pred
        df_future = pd.concat([df_future, row.to_frame().T], ignore_index=True)

        results.append({"Date": next_date, "Predicted": pred})
        last_date = next_date

    return pd.DataFrame(results)



model = CatBoostRegressor()
model.load_model("catboost_model.cbm")

# прогноз до конца сентября
df = pd.read_csv(
    'expanded.csv',
    sep=';',
    parse_dates=['Date'],
    dayfirst=False
)

# ----------------------
# Лаги и скользящие
# ----------------------
lag_days = [1,2,3,7,14,21,28,60,90]
for lag in lag_days:
    df[f'sold_lag{lag}'] = df['Sold'].shift(lag).fillna(0)

ma_windows = [3,7,14,21,28,60,90]
for w in ma_windows:
    df[f'sold_ma{w}'] = df['Sold'].rolling(w).mean().fillna(0)
    df[f'sold_median{w}'] = df['Sold'].rolling(w).median().fillna(0)

# новые фичи: волатильность
df["sold_std7"] = df["Sold"].rolling(7).std().fillna(0)
df["sold_std30"] = df["Sold"].rolling(30).std().fillna(0)

# бинарные признаки
df['sold_last_day_binary']   = (df['sold_lag1']  > 0).astype(int)
df['sold_last_week_binary']  = (df['sold_lag7']  > 0).astype(int)
df['sold_last_month_binary'] = (df['sold_lag28'] > 0).astype(int)

# ----------------------
# Новые признаки
# ----------------------
df["brand_region"] = df["Марка"].astype(str) + "_" + df["Регион"].astype(str)
df["type_flavor"]  = df["Тип"].astype(str)   + "_" + df["Вкус"].astype(str)

df["quarter"] = df["Date"].dt.quarter
df["is_start_of_month"] = (df["Date"].dt.day <= 3).astype(int)
df["is_end_of_month"]   = (df["Date"].dt.day >= 27).astype(int)

# ----------------------
# Фичи и таргет
# ----------------------
product_features = [
    "ProductGroup","ProductType","Вид","Регион","Тип",
    "Марка","Код производителя","Вкус","Ароматизированный",
    "brand_region","type_flavor"
]

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
for col in cat_features:
    df[col] = df[col].fillna("NA").astype(str)
forecast_df = forecast(model, df, features, cat_features, "2025-09-30")
print(forecast_df)

forecast_df.to_csv("future_forecast.csv", sep=";", index=False)
print("✅ Прогноз сохранён в future_forecast.csv")