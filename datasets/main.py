from etna.datasets.utils import Dataset
from etna.datasets import TSDataset
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

DATA_CSV = os.getenv("MAIN_CSV_SRC")

def load_actual_dataset() -> TSDataset:
    df = pd.read_csv(DATA_CSV)
    df = df.copy()
    ts_df = TSDataset.to_dataset(df=df)
    ts = TSDataset(ts_df, freq="D")
    return ts
