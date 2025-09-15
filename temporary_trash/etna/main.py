from etna.transforms import FilterFeaturesTransform
from etna.models.nn import DeepStateModel
from etna.models.nn.deepstate import CompositeSSM
from etna.models.nn.deepstate import LevelTrendSSM
from etna.models.nn.deepstate import SeasonalitySSM
import random
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch
from etna.models.nn import NBeatsGenericModel
from etna.models.nn import NBeatsInterpretableModel
from etna.analysis import plot_backtest
from etna.datasets.tsdataset import TSDataset
from etna.metrics import MAE, RMSE
from etna.metrics import MAPE
from etna.metrics import SMAPE
from etna.models import SeasonalMovingAverageModel
from etna.pipeline import Pipeline
from etna.transforms import DateFlagsTransform
from etna.transforms import LabelEncoderTransform
from etna.transforms import LagTransform
from etna.transforms import LinearTrendTransform
from etna.transforms import SegmentEncoderTransform
from etna.transforms import StandardScalerTransform

DATA_CSV = "expanded_etna.csv"
FORECAST_DAYS = 30
TARGET_COL = "target"
SEGMENT_COL = "segment"
OUTPUT_PLOT = "backtest_nbeats_main.png"


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# -----------------------------
# 1. Загружаем данные
# -----------------------------
df = pd.read_csv(DATA_CSV)
ts = TSDataset(df, freq="D")


HORIZON = 7
SEGMENT_FILTER = "АТ Москва|TALTHA-BP0026"  # нужный сегмент

df = pd.read_csv(DATA_CSV)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values(['segment', 'timestamp']).reset_index(drop=True)

# порог — минимум, который требуется модели
threshold = (2 * HORIZON) + HORIZON  # input_size + output_size = 3*HORIZON
print("Threshold (min required length):", threshold)

# считаем непустые значения таргета по сегменту
counts = df.dropna(subset=[TARGET_COL]).groupby(SEGMENT_COL)[TARGET_COL].count()
print(counts.describe())

# список сегментов, которые слишком короткие
bad_segments = counts[counts <= threshold].index.tolist()
if bad_segments:
    print(f"Обрезаем {len(bad_segments)} коротких сегментов:")
    for s in bad_segments:
        print("  -", s)
else:
    print("Коротких сегментов не найдено.")

# фильтруем датафрейм — оставляем только валидные сегменты
df = df[df[SEGMENT_COL] == SEGMENT_FILTER].copy()

df_filtered = df[~df[SEGMENT_COL].isin(bad_segments)].copy()

# safety-check: если вдруг все сегменты отрезало — падаем с понятной ошибкой
if df_filtered[SEGMENT_COL].nunique() == 0:
    raise RuntimeError("После фильтрации не осталось сегментов. Проверь данные или порог threshold.")

# создаём TSDataset из отфильтрованных данных
ts = TSDataset(df_filtered, freq="D")

metrics = [SMAPE(), MAPE(), MAE(), RMSE()]
model_sma = SeasonalMovingAverageModel(window=5, seasonality=7)
linear_trend_transform = LinearTrendTransform(in_column="target")

num_lags = 7

transforms = [
    SegmentEncoderTransform(),
    StandardScalerTransform(in_column="target"),
    DateFlagsTransform(
        day_number_in_week=True,
        day_number_in_month=True,
        day_number_in_year=True,
        week_number_in_month=True,
        week_number_in_year=True,
        month_number_in_year=True,
        season_number=True,
        year_number=True,
        is_weekend=True,
        out_column="dateflag",
    ),
    LagTransform(
        in_column="target",
        lags=[HORIZON + i for i in range(num_lags)],
        out_column="target_lag",
    ),
    LabelEncoderTransform(
        in_column="dateflag_day_number_in_week", strategy="none", out_column="dateflag_day_number_in_week_label"
    ),
    LabelEncoderTransform(
        in_column="dateflag_day_number_in_month", strategy="none", out_column="dateflag_day_number_in_month_label"
    ),
    FilterFeaturesTransform(exclude=["dateflag_day_number_in_week", "dateflag_day_number_in_month"]),
]


embedding_sizes = {
    "dateflag_day_number_in_week_label": (7, 7),
    "dateflag_day_number_in_month_label": (31, 7),
    "segment_code": (4, 7),
}

monthly_smm = SeasonalitySSM(num_seasons=31, timestamp_transform=lambda x: x.day - 1)
weekly_smm = SeasonalitySSM(num_seasons=7, timestamp_transform=lambda x: x.weekday())

set_seed()

model_nbeats_interp = NBeatsInterpretableModel(
    input_size=4 * HORIZON,
    output_size=HORIZON,
    loss="smape",
    trend_layer_size=64,
    seasonality_layer_size=256,
    trainer_params=dict(max_epochs=1000),
    lr=1e-3,
)

pipeline_nbeats_interp = Pipeline(
    model=model_nbeats_interp,
    horizon=HORIZON,
    transforms=[],
)

backtest_result_nbeats_interp = pipeline_nbeats_interp.backtest(ts, metrics=metrics, n_folds=3, n_jobs=1)

metrics_nbeats_interp = backtest_result_nbeats_interp["metrics"]
forecast_ts_list_nbeats_interp = backtest_result_nbeats_interp["forecasts"]

print(metrics_nbeats_interp)
score = metrics_nbeats_interp["SMAPE"].mean()
print(f"Average SMAPE: {score:.3f}")

fig, ax = plt.subplots(figsize=(20, 8))
plot_backtest(forecast_ts_list_nbeats_interp, ts, history_len=30)
plt.title(f"Backtest N-BEATS ({SEGMENT_FILTER})")
plt.tight_layout()
plt.savefig(OUTPUT_PLOT)
plt.close()
print(f"График сохранён в {OUTPUT_PLOT}")
