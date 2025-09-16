import pandas as pd
from etna.datasets import TSDataset
import os
from dotenv import load_dotenv

load_dotenv()

DATA_CSV = os.getenv("MAIN_CSV_SRC")
# Читаем CSV

def format_dataset():
    df = pd.read_csv(DATA_CSV, parse_dates=['Date'], dayfirst=True)
    # Преобразуем колонку Date
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
    df['segment'] = df['Department'] + '|' + df['Article']
    #df.drop(["Unnamed: 10"], axis=1, inplace=True)
    df = df.rename(columns={"Date": "timestamp", "Sold": "target"})

    # Записываем обратно в тот же файл
    df.to_csv(DATA_CSV, index=False)

    print("Готово, даты преобразованы.")

format_dataset()
