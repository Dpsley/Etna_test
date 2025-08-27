# Создам расширенный датасет с фичами (лаги, скользящие средние, разобранные свойства товаров, праздники/выходные и признаки рестока).
# Входные данные — тот самый CSV (включая сгенерированные строки до 31.01.2024).
# Сохраню файл /mnt/data/expanded_jan2024.csv и покажу первые 20 строк.

import pandas as pd
import json
from io import StringIO

# загрузим
df = pd.read_csv("/mnt/d/OSPanel/domains/Etna_test/actualdata/data2.csv", sep=',', dayfirst=True, parse_dates=['Date'])

# распарсим ProductProperties JSON в колонки
def parse_props(s):
    try:
        return json.loads(s)
    except Exception:
        # попытка заменить неправильные кавычки
        try:
            s2 = s.replace("“", '"').replace("”", '"').replace("«", '"').replace("»", '"')
            return json.loads(s2)
        except Exception:
            return {}

props_df = df['ProductProperties'].apply(lambda s: pd.Series(parse_props(s)))
# стандартизируем имена колонок: русские названия оставляем как есть
df = pd.concat([df, props_df], axis=1)

# календарные признаки
df['day_of_week'] = df['Date'].dt.weekday  # 0=Mon
df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)
df['month'] = df['Date'].dt.month
df['day'] = df['Date'].dt.day

# сорим и делаем лаги/скользящие
df = df.sort_values(['Article','Date']).reset_index(drop=True)
group = df.groupby('Article', as_index=False, sort=False)

# лаги для Sold
# --- PATCH START: исправленная секция лагов/роулингов/рестоков ---
# приводим типы и сортируем (важно)
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
df['Sold'] = pd.to_numeric(df['Sold'], errors='coerce').fillna(0).astype(float)
df['Stock'] = pd.to_numeric(df['Stock'], errors='coerce').fillna(0).astype(float)

df = df.sort_values(['Article', 'Date']).reset_index(drop=True)

# лаги для Sold (без reset_index, через groupby.shift)
df['sold_lag1']  = df.groupby('Article')['Sold'].shift(1).fillna(0)
df['sold_lag7']  = df.groupby('Article')['Sold'].shift(7).fillna(0)
df['sold_lag14'] = df.groupby('Article')['Sold'].shift(14).fillna(0)
df['sold_lag28'] = df.groupby('Article')['Sold'].shift(28).fillna(0)

# скользящие средние — через transform (гарантированно Series той же длины)
df['sold_ma7']  = df.groupby('Article')['Sold'].transform(lambda x: x.rolling(7,  min_periods=1).mean())
df['sold_ma14'] = df.groupby('Article')['Sold'].transform(lambda x: x.rolling(14, min_periods=1).mean())

# лаги для Stock и признаки рестока
df['stock_lag1'] = df.groupby('Article')['Stock'].shift(1)
df['stock_lag1'] = df['stock_lag1'].fillna(df['Stock'])  # для первого дня оставим текущее (можно заменить на NaN)
df['stock_diff'] = df['Stock'] - df['stock_lag1']
df['restock_flag'] = (df['stock_diff'] > 0).astype(int)

# days_since_last_restock: аккуратно возвращаем Series с тем же индексом
def days_since_last_restock(g):
    last = None
    out = []
    for idx, row in g.iterrows():
        if row.get('restock_flag', 0) == 1:
            last = row['Date']
            out.append(0)
        else:
            out.append(None if last is None else (row['Date'] - last).days)
    return pd.Series(out, index=g.index)

df['days_since_last_restock'] = None

for article, g in df.groupby('Article', sort=False):
    last = None
    # g уже отсортирована по Date, т.к. ранее мы отсортировали df
    for idx, row in g.iterrows():
        if row.get('restock_flag', 0) == 1:
            last = row['Date']
            df.at[idx, 'days_since_last_restock'] = 0
        else:
            df.at[idx, 'days_since_last_restock'] = None if last is None else (row['Date'] - last).days

# Сохраняем в текущую директорию (Windows-compatible) — поменяй путь, если хочешь в конкретную папку
import os
out_path = os.path.join(os.getcwd(), 'expanded.csv.csv')  # -> C:\OSPanel\domains\Etna_test\expanded_jan2024.csv
df.to_csv(out_path, index=False, sep=';')
print(f"Saved: {out_path}")

# покажем первые 20 строк
import caas_jupyter_tools as cjt
cjt.display_dataframe_to_user("expanded_jan2024_preview", df.head(20))

out_path

