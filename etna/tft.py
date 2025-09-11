# tft_from_catboost_rewrite.py
import os
import platform
import multiprocessing
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pandas.errors import SettingWithCopyWarning
from sklearn.preprocessing import LabelEncoder

warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)

import optuna
import torch

from etna.datasets import TSDataset
from etna.pipeline import Pipeline
from etna.models.nn import TFTModel
from etna.transforms import (
    TimeSeriesImputerTransform, DateFlagsTransform, HolidayTransform,
    FourierTransform, SegmentEncoderTransform, LogTransform
)
from etna.metrics import RMSE

# ---- Настройки ----
DATA_CSV = "sales_remains_072023_062025.csv"
FORECAST_DAYS = 30
OUTPUT_PLOT = "auto_backtest.png"
SEGMENT_NAME = 'АТ Москва|CIMPCH-000062'
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # если не используешь GPU

def safe_num_workers(default=0):
    if platform.system() == "Windows":
        return 0
    try:
        return min(4, max(0, (os.cpu_count() or 1) - 1))
    except Exception:
        return default

NUM_WORKERS = safe_num_workers()

def load_and_prepare(path=DATA_CSV, segment_name=SEGMENT_NAME):
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    # удаляем лишние колонки (если есть)
    drop_cols = ["ProductName", "ProductType", "Article", "Department", "ProductProperties", "ProductGroup", "Stock"]
    for c in drop_cols:
        if c in df.columns:
            df.drop(c, axis=1, inplace=True)

    # фильтрация по сегменту
    df = df[df["segment"] == segment_name].copy()
    df.reset_index(drop=True, inplace=True)

    # явный integer для статической категории (чтобы не получить category в TSDataset)
    le = LabelEncoder()
    df["segment_code"] = le.fit_transform(df["segment"]).astype(np.int32)

    ts_df = TSDataset.to_dataset(df=df)
    ts_raw = TSDataset(ts_df, freq="D")
    return df, ts_raw, {"segment_code": int(df["segment_code"].nunique())}

def build_transforms():
    return [
        TimeSeriesImputerTransform(in_column="target"),  # важно: уменьшает NaN в истории
        LogTransform(in_column="target", out_column="target_log", inplace=False),  # опционально
        SegmentEncoderTransform(),  # создаёт удобную статическую кодировку
        DateFlagsTransform(
            day_number_in_week=True, day_number_in_month=True, day_number_in_year=True,
            week_number_in_month=True, week_number_in_year=True, month_number_in_year=True,
            year_number=True, is_weekend=True, out_column="flag"
        ),
        HolidayTransform(out_column="holiday", iso_code="RUS"),
        FourierTransform(in_column="target", period=31, order=2, out_column="fourier_31"),
    ]

def detect_features_example(df, transforms):
    ts_tmp = TSDataset(TSDataset.to_dataset(df=df), freq="D")
    for tr in transforms:
        tr.fit(ts_tmp)
        ts_tmp = tr.transform(ts_tmp)
    transformed_flat = ts_tmp.to_pandas(flatten=True)
    return transformed_flat

def build_feature_lists(transformed_flat):
    exclude = {"timestamp", "segment", "target", "segment_code", "index"}
    cand = [c for c in transformed_flat.columns if c not in exclude]
    time_varying_reals_encoder = [c for c in cand if transformed_flat[c].dtype.kind in "fiu"]
    time_varying_reals_decoder = ["target"]
    static_categoricals = ["segment_code"]
    static_reals = []
    return time_varying_reals_encoder, time_varying_reals_decoder, static_categoricals, static_reals

def fit_and_forecast(df, ts_raw, train_ts, transforms, tv_enc, tv_dec, static_cat, static_reals, num_embeddings):
    params = dict(
        encoder_length=30,
        decoder_length=FORECAST_DAYS,
        n_heads=2,
        num_layers=1,
        hidden_size=64,
        dropout=0.1,
        static_categoricals=static_cat,
        static_reals=static_reals,
        time_varying_categoricals_encoder=[],
        time_varying_categoricals_decoder=[],
        time_varying_reals_encoder=tv_enc,
        time_varying_reals_decoder=tv_dec,
        num_embeddings=num_embeddings,
        train_batch_size=32,
        test_batch_size=32,
        optimizer_params={},
        train_dataloader_params={"num_workers": NUM_WORKERS},
        val_dataloader_params={"num_workers": NUM_WORKERS},
        trainer_params={"max_epochs": 8, "accelerator": "cpu", "devices": 1},
    )

    model = TFTModel(**params)
    pipeline = Pipeline(model=model, transforms=transforms, horizon=FORECAST_DAYS)

    # обучаем на train_ts — transforms применятся внутри pipeline.fit
    pipeline.fit(train_ts)

    # Forecast: передаём fresh TSDataset, созданный из исходного df,
    # чтобы избежать несоответствия колонок при make_future
    ts_for_forecast = TSDataset(TSDataset.to_dataset(df=df), freq="D")
    forecast = pipeline.forecast(ts=ts_for_forecast, prediction_interval=False)

    # проверка на NaN в прогнозе
    f_df = forecast.to_pandas(flatten=True).reset_index()
    if f_df["target"].isna().any():
        print("В прогнозе есть NaN в target — количество:", int(f_df["target"].isna().sum()))
    return pipeline, forecast

def plot_and_report(test_ts, forecast, output_plot=OUTPUT_PLOT):
    test_df = test_ts.to_pandas(flatten=True).reset_index()
    forecast_df = forecast.to_pandas(flatten=True).reset_index()
    test_df["timestamp"] = pd.to_datetime(test_df["timestamp"])
    forecast_df["timestamp"] = pd.to_datetime(forecast_df["timestamp"])
    test_df["article"] = test_df["segment"].str.split("|").str[0]
    forecast_df["article"] = forecast_df["segment"].str.split("|").str[0]

    test_agg = test_df.groupby(["timestamp", "article"], as_index=False)["target"].sum()
    forecast_agg = forecast_df.groupby(["timestamp", "article"], as_index=False)["target"].sum()

    total_fact = test_agg.groupby("timestamp", as_index=False)["target"].sum()
    total_forecast = forecast_agg.groupby("timestamp", as_index=False)["target"].sum()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(total_fact["timestamp"], total_fact["target"], label="Actual (факт)", linewidth=2)
    ax.plot(total_forecast["timestamp"], total_forecast["target"], label="Forecast (план)", linestyle="--", linewidth=2)
    ax.set_title("Forecast vs Actual (сумма по article)")
    ax.set_xlabel("Date"); ax.set_ylabel("Target"); ax.legend()
    plt.tight_layout(); plt.savefig(output_plot, dpi=150); plt.close()
    print("Saved plot to", os.path.abspath(output_plot))
    print("Сумма факта за период: %.2f" % total_fact["target"].sum())
    print("Сумма прогноза за период: %.2f" % total_forecast["target"].sum())
    print("Разница (прогноз - факт): %.2f" % (total_forecast["target"].sum() - total_fact["target"].sum()))

def main():
    if platform.system() == "Windows":
        multiprocessing.freeze_support()

    df, ts_raw, num_embeddings = load_and_prepare(DATA_CSV, SEGMENT_NAME)
    print("Loaded rows:", len(df))

    transforms = build_transforms()
    transformed_flat = detect_features_example(df, transforms)
    print("Sample transformed cols:", transformed_flat.columns.tolist()[:60])

    tv_enc, tv_dec, static_cat, static_reals = build_feature_lists(transformed_flat)
    print("Detected encoder features:", tv_enc)

    train_ts, test_ts = ts_raw.train_test_split(
        train_start="2023-07-01", train_end="2025-05-31",
        test_start="2025-06-01", test_end="2025-06-30"
    )

    pipeline, forecast = fit_and_forecast(df, ts_raw, train_ts, transforms, tv_enc, tv_dec, static_cat, static_reals, num_embeddings)
    plot_and_report(test_ts, forecast)

if __name__ == "__main__":
    main()
