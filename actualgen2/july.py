import pandas as pd
import numpy as np
import json
from catboost import CatBoostRegressor, Pool


# Загружаем историю
history = pd.read_csv(
    'expanded.csv',
    sep=';',
    parse_dates=['Date'],
    dayfirst=False
)

# Берём уникальные товары
unique_products = history.drop_duplicates(subset=["Department", "Article"])[
    ['Department', 'Article', "ProductName", "ProductGroup", "ProductType", "ProductProperties"]
]


def parse_props(s):
    try:
        s2 = s.replace("“", '"').replace("”", '"').replace("«", '"').replace("»", '"')
        props = json.loads(s2)
        for k, v in props.items():
            if v == "" or v is None:
                props[k] = "Unset"
        return props
    except Exception:
        return {}

def safe_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero = y_true != 0
    if not non_zero.any():
        return float("inf")
    return np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100

def expand_products_by_date(unique_products, start_date: str, end_date: str) -> pd.DataFrame:
    dates = pd.date_range(start=start_date, end=end_date)
    expanded = unique_products.merge(pd.DataFrame({'Date': dates}), how='cross')
    expanded['Sold'] = 0
    expanded['Stock'] = 1
    expanded = expanded.sort_values('Date').reset_index(drop=True)
    return expanded

def create_lags_and_forecast(df):
    props_df = df['ProductProperties'].apply(lambda s: pd.Series(parse_props(s)))
    #return props_df
    props_df = props_df[[c for c in props_df.columns if c not in df.columns]]

    df = pd.concat([df, props_df], axis=1)

    # Календарные признаки
    df['day_of_week'] = df['Date'].dt.weekday
    df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)
    df['month'] = df['Date'].dt.month
    df['day'] = df['Date'].dt.day
    df["brand_region"] = df["Марка"].astype(str) + "_" + df["Регион"].astype(str)
    df["type_flavor"]  = df["Тип"].astype(str)   + "_" + df["Вкус"].astype(str)

    df["quarter"] = df["Date"].dt.quarter
    df["is_start_of_month"] = (df["Date"].dt.day <= 3).astype(int)
    df["is_end_of_month"]   = (df["Date"].dt.day >= 27).astype(int)
    # Приводим типы
    for col in ['Sold']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True)

    # Сортировка
    df = df.sort_values(['Department','Article','Date']).reset_index(drop=True)

    # Лаги и скользящие средние по (Department, Article)
    group_cols = ['Department','Article']

    lag_days = [1,2,3,7,14,21,28,60,90]
    for lag in lag_days:
        df[f'sold_lag{lag}'] = df.groupby(group_cols)['Sold'].shift(lag).fillna(0)

    sold_std = [7, 30]
    for std_step in sold_std:
        df[f'sold_std{std_step}'] = df.groupby(group_cols)['Sold'].transform(lambda x: x.shift(1).rolling(std_step, min_periods=1).std().fillna(0))

    df['sold_last_day_binary']   = (df['sold_lag1']  > 0).astype(int)
    df['sold_last_week_binary']  = (df['sold_lag7']  > 0).astype(int)
    df['sold_last_month_binary'] = (df['sold_lag28'] > 0).astype(int)

    ma_windows = [3,7,14,21,28,60,90]
    for w in ma_windows:
        df[f'sold_ma{w}'] = df.groupby(group_cols)['Sold'].transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean().fillna(0))
        df[f'sold_median{w}'] = df.groupby(group_cols)['Sold'].transform(lambda x: x.shift(1).rolling(w, min_periods=1).median().fillna(0))
    return df

def predictor(df):
    df.to_csv("lags.csv", index=False)

    product_features = [
        "ProductGroup", "ProductType", "Вид", "Регион", "Тип",
        "Марка", "Код производителя", "Вкус", "Ароматизированный",
        "brand_region", "type_flavor"
    ]
    lag_days = [1, 2, 3, 7, 14, 21, 28, 60, 90]
    sold_std = [7, 30]
    ma_windows = [3, 7, 14, 21, 28, 60, 90]

    features = product_features + [
        "Department",
        *[f'sold_lag{l}' for l in lag_days],
        *[f'sold_ma{w}' for w in ma_windows],
        *[f'sold_median{w}' for w in ma_windows],
        *[f'sold_std{d}' for d in sold_std],
        "sold_last_day_binary", "sold_last_week_binary", "sold_last_month_binary",
        "day_of_week", "is_weekend", "month", "day",
        "quarter", "is_start_of_month", "is_end_of_month"
    ]
    cat_features = ["Department"] + product_features
    for col in cat_features:
        df[col] = df[col].fillna("NA").astype(str)

    df['Sold'] = pd.to_numeric(df['Sold'], errors='coerce').fillna(0).astype(int)

    df['Sold'] = df['Sold'].clip(lower=0)

    model = CatBoostRegressor()
    model.load_model("catboost_model.cbm", format="cbm")

    X = df[features].copy()
    print("X", X)
    pool = Pool(data=X, cat_features=cat_features)
    preds = model.predict(pool)
    df['Sold'] = np.expm1(preds).round().astype(int)
    print("result", np.expm1(preds).round().astype(int))

# пример
#july_df = expand_products_by_date(unique_products, '2025-07-01', '2025-07-01')
#print(july_df)
#    # превращаем Series в DataFrame, чтобы concat работал правильно
#history = pd.concat([history, july_df])
#print(history)
#history = create_lags_and_forecast(history)
#print(history)
#predictor(history)
#history.to_csv('july.csv', index=False)

future_dates = pd.date_range("2025-07-01", "2025-07-31")

for current_date in future_dates:
    # 1. расширяем на один день
    next_day_df = expand_products_by_date(unique_products, current_date, current_date)
    print("next_day_df", next_day_df)

    next_day_props = next_day_df['ProductProperties'].apply(lambda s: pd.Series(parse_props(s)))
    next_day_df = pd.concat([next_day_df, next_day_props], axis=1)

    # убедимся, что все колонки совпадают с history
    for col in history.columns:
        if col not in next_day_df.columns:
            next_day_df[col] = np.nan

    # 2. добавляем к истории только для расчета лагов
    print("next_day_df2", next_day_df)
    print("history", history)

    temp_df = pd.concat([history, next_day_df], ignore_index=True)
    print("temp_df", temp_df)
    temp_df = temp_df.sort_values(['Department', 'Article', 'Date']).reset_index(drop=True)   # 3. пересчитываем лаги только для нового дня
    temp_df = create_lags_and_forecast(temp_df)
    # 4. выбираем только прогнозный день
    next_day_with_lags = temp_df[temp_df['Date'] == current_date].copy()

    # 5. предсказываем
    predictor(next_day_with_lags)

    # 6. добавляем прогноз в историю
    next_day_with_lags['Sold'] = next_day_with_lags['Sold'].round().astype(int)
    history = pd.concat([history, next_day_with_lags], ignore_index=True)
history.to_csv('july.csv', index=False)