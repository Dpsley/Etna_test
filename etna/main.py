# forecast_etna_nbeats_gpu_fixed.py
import pandas as pd
import json
from etna.datasets import TSDataset
from etna.models.nn import NBeatsInterpretableModel
from etna.transforms import LagTransform, DateFlagsTransform, LabelEncoderTransform
from etna.pipeline import Pipeline
import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--input", default="expanded_etna.csv")
parser.add_argument("--forecast_days", type=int, default=30)
parser.add_argument("--output", default="forecast.csv")
parser.add_argument("--department", default=None)
parser.add_argument("--article", default=None)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Используем устройство: {device}")

df = pd.read_csv(args.input, parse_dates=["timestamp"])
df.rename(columns={"Sold": "target"}, inplace=True)

if args.department:
    df = df[df["Department"] == args.department].copy()
if args.article:
    df = df[df["Article"] == args.article].copy()

def parse_props(s):
    try:
        return json.loads(s)
    except:
        return {}

props_df = df["ProductProperties"].apply(parse_props).apply(pd.Series)
df = pd.concat([df, props_df], axis=1)

df["segment"] = df["Department"].astype(str) + "|" + df["Article"].astype(str)

ts_df = df[["timestamp", "segment", "target", "ProductType"] + list(props_df.columns)]
ts = TSDataset(ts_df, freq="D")

transforms = [
    LagTransform(in_column="target", lags=[1, 2, 3, 7, 14, 21, 30, 45, 60, 75, 90, 180]),
    DateFlagsTransform(
        day_number_in_week=True,
        day_number_in_month=True,
        month_number_in_year=True,
        week_number_in_month=True,
        week_number_in_year=True,
        season_number=True,
        year_number=True,
        is_weekend=True
    ),
]

cat_cols = ["ProductType"] + list(props_df.columns)
for col in cat_cols:
    if col in df.columns:
        transforms.append(LabelEncoderTransform(in_column=col))

model = NBeatsInterpretableModel(
    input_size=30,
    output_size=args.forecast_days,
    loss="smape",
    trend_blocks=3,
    trend_layers=4,
    trend_layer_size=256,
    degree_of_polynomial=2,
    seasonality_blocks=3,
    seasonality_layers=4,
    seasonality_layer_size=2048,
    num_of_harmonics=1,
    lr=1e-3,
    train_batch_size=1024,
    test_batch_size=1024,
    trainer_params={"max_epochs": 50, "accelerator": "gpu" if device=="cuda" else "cpu", "devices": 1}
)

pipeline = Pipeline(model=model, transforms=transforms, horizon=args.forecast_days)

print("Обучаем NBEATS модель...")
pipeline.fit(ts)

print(f"Делаем прогноз на {args.forecast_days} дней...")
forecast = pipeline.forecast()

forecast_df = forecast.to_pandas().reset_index()
forecast_df.rename(columns={"timestamp": "Date"}, inplace=True)
forecast_df.to_csv(args.output, index=False)
print(f"Прогноз сохранён в {args.output}")
