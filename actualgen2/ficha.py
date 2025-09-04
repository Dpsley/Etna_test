# predict_recursive.py

import pandas as pd
import numpy as np
import json
from catboost import CatBoostRegressor, Pool

# ---------------------------
# Настройки
# ---------------------------
DATA_CSV = "data.csv"
MODEL_PATH = "catboost_model.cbm"

# ---------------------------
# Загрузка модели
# ---------------------------
model = CatBoostRegressor()
model.load_model(MODEL_PATH)

# ---------------------------
# Загрузка и подготовка данных
# ---------------------------
df = pd.read_csv(DATA_CSV, sep=',', dayfirst=True, parse_dates=['Date'])

def parse_props(s):
    try:
        return json.loads(s)
    except Exception:
        try:
            s2 = s.replace("“", '"').replace("”", '"').replace("«", '"').replace("»", '"')
            return json.loads(s2)
        except Exception:
            return {}

props_df = df['ProductProperties'].apply(lambda s: pd.Series(parse_props(s)))
df = pd.concat([df, props_df], axis=1)

df['Sold'] = pd.to_numeric(df['Sold'], errors='coerce').fillna(0)
df['Date'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True)
df = df.sort_values(['Department','Article','Date']).reset_index(drop=True)

# Календарные признаки
df['day_of_week'] = df['Date'].dt.weekday
df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)
df['month'] = df['Date'].dt.month
df['day'] = df['Date'].dt.day
df['quarter'] = df['Date'].dt.quarter
df['is_start_of_month'] = (df['Date'].dt.day <= 3).astype(int)
df['is_end_of_month'] = (df['Date'].dt.day >= 27).astype(int)

# ---------------------------
# Лаги и скользящие средние
# ---------------------------
lag_days = [1,2,3,7,14,21,28,60,90]
ma_windows = [3,7,14,21,28,60,90]

def build_features_recursive(article_code, start_date, n_days):
    """
    Рекурсивный прогноз на n_days вперед
    """
    df_article = df[df['Article'] == article_code].copy()
    df_article = df_article.sort_values('Date').reset_index(drop=True)

    # Создаём копию для прогноза
    last_date = df_article['Date'].max()
    future_dates = pd.date_range(start=max(last_date, pd.to_datetime(start_date, dayfirst=True)) + pd.Timedelta(days=1),
                                 periods=n_days)
    forecast_df = pd.DataFrame({'Date': future_dates})
    forecast_df['Article'] = article_code
    forecast_df['Department'] = df_article['Department'].iloc[0]

    df_combined = pd.concat([df_article, forecast_df], ignore_index=True, sort=False)
    df_combined = df_combined.sort_values('Date').reset_index(drop=True)

    # Календарные признаки для всех строк
    df_combined['day_of_week'] = df_combined['Date'].dt.weekday
    df_combined['is_weekend'] = df_combined['day_of_week'].isin([5,6]).astype(int)
    df_combined['month'] = df_combined['Date'].dt.month
    df_combined['day'] = df_combined['Date'].dt.day
    df_combined['quarter'] = df_combined['Date'].dt.quarter
    df_combined['is_start_of_month'] = (df_combined['Date'].dt.day <= 3).astype(int)
    df_combined['is_end_of_month'] = (df_combined['Date'].dt.day >= 27).astype(int)

    # Комбинированные фичи
    df_combined['brand_region'] = df_combined['Марка'].astype(str) + "_" + df_combined['Регион'].astype(str)
    df_combined['type_flavor']  = df_combined['Тип'].astype(str)   + "_" + df_combined['Вкус'].astype(str)

    # Рекурсивное заполнение лагов и скользящих
    for i, row in df_combined.iterrows():
        if pd.notna(row['Sold']):
            continue  # Исторические данные остаются
        for lag in lag_days:
            lag_val = df_combined.loc[max(0,i-lag),'Sold'] if i-lag >=0 else 0
            df_combined.at[i,f'sold_lag{lag}'] = lag_val
        for w in ma_windows:
            window = df_combined['Sold'].iloc[max(0,i-w):i]
            df_combined.at[i,f'sold_ma{w}'] = window.mean()
            df_combined.at[i,f'sold_median{w}'] = window.median()
        # Волатильность
        df_combined.at[i,'sold_std7']  = df_combined['Sold'].iloc[max(0,i-7):i].std() or 0
        df_combined.at[i,'sold_std30'] = df_combined['Sold'].iloc[max(0,i-30):i].std() or 0
        # Бинарные признаки
        df_combined.at[i,'sold_last_day_binary']   = int(df_combined.at[i,'sold_lag1'] > 0)
        df_combined.at[i,'sold_last_week_binary']  = int(df_combined.at[i,'sold_lag7'] > 0)
        df_combined.at[i,'sold_last_month_binary'] = int(df_combined.at[i,'sold_lag28'] > 0)

        # Прогноз
        feature_cols = [
            "ProductGroup","ProductType","Вид","Регион","Тип",
            "Марка","Код производителя","Вкус","Ароматизированный",
            "brand_region","type_flavor",
            "Department",
            *[f'sold_lag{l}' for l in lag_days],
            *[f'sold_ma{w}' for w in ma_windows],
            *[f'sold_median{w}' for w in ma_windows],
            "sold_std7","sold_std30",
            "sold_last_day_binary","sold_last_week_binary","sold_last_month_binary",
            "day_of_week","is_weekend","month","day",
            "quarter","is_start_of_month","is_end_of_month"
        ]
        # Приведение категориальных к str
        cat_cols = ["Department","ProductGroup","ProductType","Вид","Регион","Тип",
                    "Марка","Код производителя","Вкус","Ароматизированный",
                    "brand_region","type_flavor"]
        for c in cat_cols:
            if c in df_combined.columns:
                df_combined.at[i,c] = str(df_combined.at[i,c])

        X_row = df_combined.loc[[i], feature_cols]
        pred = model.predict(X_row)
        df_combined.at[i,'Sold'] = max(0,int(round(pred[0])))

    return df_combined[df_combined['Date'] >= pd.to_datetime(start_date, dayfirst=True)][['Date','Sold']]

# ---------------------------
# Пример использования
# ---------------------------
if __name__ == "__main__":
    article = "TALTHA-BP0026"
    start_date = "01.07.2025"
    n_days = 3

    forecast = build_features_recursive(article, start_date, n_days)
    print(forecast)
