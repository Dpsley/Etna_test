#normalize.py

import pandas as pd
import json
import os

# Загрузка CSV
df = pd.read_csv("/mnt/d/OSPanel/domains/Etna_test/actualgen2/sales_remains_072023_062025.csv",
                 sep=',', dayfirst=True, parse_dates=['Date'])

print(df.columns.tolist())

# Парсинг JSON ProductProperties
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

props_df = df['ProductProperties'].apply(lambda s: pd.Series(parse_props(s)))

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

for col in ['Reserve','Available']:
    if col in df.columns:
        df.drop(columns=[col], inplace=True)

# Сохраняем расширенный CSV
out_path = os.path.join(os.getcwd(), 'expanded.csv')
df.to_csv(out_path, sep=';', index=False)
print(f"✅ Saved: {out_path}")
