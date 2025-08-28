import pandas as pd
import json
import os

# Загрузка CSV
df = pd.read_csv("/mnt/d/OSPanel/domains/Etna_test/actualdata/sales_full.csv",
                 sep=',', dayfirst=True, parse_dates=['Date'])

# Парсинг JSON ProductProperties
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

# Календарные признаки
df['day_of_week'] = df['Date'].dt.weekday
df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)
df['month'] = df['Date'].dt.month
df['day'] = df['Date'].dt.day

# Приводим типы
for col in ['Sold','Stock','Reserve','Available']:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

df['Date'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True)

# Сортировка
df = df.sort_values(['Department','Article','Date']).reset_index(drop=True)

# Лаги и скользящие средние по (Department, Article)
group_cols = ['Department','Article']

df['sold_lag1']  = df.groupby(group_cols)['Sold'].shift(1).fillna(0)
df['sold_lag7']  = df.groupby(group_cols)['Sold'].shift(7).fillna(0)
df['sold_lag14'] = df.groupby(group_cols)['Sold'].shift(14).fillna(0)
df['sold_lag28'] = df.groupby(group_cols)['Sold'].shift(28).fillna(0)

df['sold_ma7']  = df.groupby(group_cols)['Sold'].transform(lambda x: x.rolling(7,min_periods=1).mean())
df['sold_ma14'] = df.groupby(group_cols)['Sold'].transform(lambda x: x.rolling(14,min_periods=1).mean())

# Stock лаги и рестоки
df['stock_lag1'] = df.groupby(group_cols)['Stock'].shift(1).fillna(df['Stock'])
df['stock_diff'] = df['Stock'] - df['stock_lag1']
df['restock_flag'] = (df['stock_diff'] > 0).astype(int)

# Days since last restock
df['days_since_last_restock'] = None
for (dept, article), g in df.groupby(group_cols, sort=False):
    last = None
    for idx, row in g.iterrows():
        if row['restock_flag'] == 1:
            last = row['Date']
            df.at[idx, 'days_since_last_restock'] = 0
        else:
            df.at[idx, 'days_since_last_restock'] = None if last is None else (row['Date'] - last).days

# Сохраняем расширенный CSV
out_path = os.path.join(os.getcwd(), 'expanded.csv')
df.to_csv(out_path, sep=';', index=False)
print(f"✅ Saved: {out_path}")
