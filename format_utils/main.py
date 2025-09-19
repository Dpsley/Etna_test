import pandas as pd
from etna.datasets import TSDataset
import os
import json
from dotenv import load_dotenv

load_dotenv()

DATA_CSV = os.getenv("MAIN_CSV_SRC")
# Читаем CSV

def format_dataset():
    df = pd.read_csv(DATA_CSV)

    #df = pd.read_csv(DATA_CSV, parse_dates=['Date'], dayfirst=True)
    # Преобразуем колонку Date
    #df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
    df['segment'] = df['Department'] + '|' + df['Article']
    #df.drop(["Unnamed: 10"], axis=1, inplace=True)
    #df = df.rename(columns={"Date": "timestamp", "Sold": "target"})
    df["ProductProperties"] = df["ProductProperties"].apply(json.loads)

    # разваливаем словари в колонки
    props_df = pd.json_normalize(df["ProductProperties"])
    props_df = props_df.add_prefix("prop_")  # чтобы не путалось

    df.drop(["Article", "ProductProperties"], axis=1,
            inplace=True)
    df = pd.concat([df, props_df], axis=1)
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].replace('', 'unknown')  # вот это главное
            df[col] = df[col].fillna("unknown")
        else:
            df[col] = df[col].fillna(0)

    df.to_csv(DATA_CSV, index=False)
    # Записываем обратно в тот же файл

    print("Готово, даты преобразованы.")

format_dataset()
