from etna.datasets.utils import Dataset
from etna.datasets import TSDataset
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

DATA_CSV = os.getenv("MAIN_CSV_SRC")

def load_actual_dataset() -> TSDataset:
    df = pd.read_csv(DATA_CSV)
    print("Исходные колонки:", df.columns.tolist())

    # убираем все колонки, которые начинаются с 'prop_'
    #cols_to_drop = [col for col in df.columns if col.startswith("prop_") or col.startswith("productName") or col.startswith("article")]
    #df = df.drop(columns=cols_to_drop)
    #print("Колонки после удаления prop_*:", df.columns.tolist())
    segments_with_non_zero = df.groupby('segment')['target'].apply(lambda x: (x != 0).any())
    segments_to_keep = segments_with_non_zero[segments_with_non_zero].index.tolist()
    df = df[df['segment'].isin(segments_to_keep)]
    # создаём TSDataset
    ts_df = TSDataset.to_dataset(df=df)
    ts = TSDataset(ts_df, freq="D")
    return ts

