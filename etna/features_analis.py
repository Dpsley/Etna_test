import pandas as pd
from etna.datasets import TSDataset
from etna.pipeline import Pipeline
from etna.models import CatBoostMultiSegmentModel
from etna.transforms import (
    LagTransform, DateFlagsTransform, SegmentEncoderTransform, LinearTrendTransform,
    STLTransform, TimeSeriesImputerTransform, FourierTransform, HolidayTransform,
    MeanTransform, LogTransform, StdTransform
)
from etna.metrics import RMSE
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import optuna
import random
import numpy as np
import warnings
from pandas.errors import SettingWithCopyWarning
warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)

# -----------------------------
# Настройки
# -----------------------------
DATA_CSV = "sales_remains_072023_062025.csv"
FORECAST_DAYS = 30
OUTPUT_PLOT = "auto_backtest.png"
segment_name = 'АТ Москва|CIMPCH-000062'

os.environ["CUDA_VISIBLE_DEVICES"] = ""

# -----------------------------
# 1. Загружаем данные
# -----------------------------
df = pd.read_csv(DATA_CSV)
np.random.seed(42)
random.seed(42)
#print("=== Исходный DF ===")
#print(df.head())
#print(df.info())
df.drop(["ProductName", "ProductType", "Article", "Department", "ProductProperties", "ProductGroup", "Stock"], axis=1, inplace=True)
#df = df.copy()
df = df[df['segment'] == segment_name].copy()

ts_df = TSDataset.to_dataset(df=df)
print("=== TSDataset.to_dataset ===")
print(ts_df.head())
print(ts_df.info())
ts = TSDataset(ts_df, freq="D")
ts.head(5)
#exit(-9)
train_ts, test_ts = ts.train_test_split(train_start="2023-07-01", train_end="2025-05-31", test_start="2025-06-01", test_end="2025-06-30")
print("=== Сегменты TSDataset ===")
print(ts.segments)
print("=== Train/ Test shapes ===")
#print(train_ts[:, :, "target"].shape, test_ts[:, :, "target"].shape)
# -----------------------------
# 2. Трансформеры
# -----------------------------
seg = SegmentEncoderTransform()
lags_small = LagTransform(in_column="target", lags=[21,24,27,28,29,31], out_column="target_lag_small")
lags = LagTransform(in_column="target", lags=[31], out_column="target_lag")
lags_macro = LagTransform(in_column="target", lags=[1,3,5,7], out_column="lags_macro")
lags_macro_2 = LagTransform(in_column="target", lags=[2,4,6], out_column="lags_macro_2")
stl = STLTransform(in_column="target", period=7)
trend = LinearTrendTransform(in_column="target")
imputer = TimeSeriesImputerTransform(in_column="target")
mean_tr_1 = MeanTransform(in_column="target", alpha=0.5, seasonality=31, min_periods=1, out_column="mean_1", window=1)
mean_tr_3 = MeanTransform(in_column="target", alpha=0.5, seasonality=31, min_periods=3, out_column="mean_3", window=3)
mean_tr_7 = MeanTransform(in_column="target", alpha=0.5, seasonality=31, min_periods=7, out_column="mean_7", window=7)
mean_tr_14 = MeanTransform(in_column="target", alpha=0.5, seasonality=31, min_periods=14, out_column="mean_14", window=14)
mean_tr_30 = MeanTransform(in_column="target", alpha=0.5, seasonality=31, min_periods=30, out_column="mean_30", window=30)
mean_tr_60 = MeanTransform(in_column="target", alpha=0.5, seasonality=31, min_periods=60, out_column="mean_60", window=60)

std_1 = StdTransform(in_column="target", seasonality=31, window=1, min_periods=1, out_column="std_1", ddof=1)
std_3 = StdTransform(in_column="target", seasonality=31, window=3, min_periods=3, out_column="std_3", ddof=1)
std_7 = StdTransform(in_column="target", seasonality=31, window=7, min_periods=7, out_column="std_7", ddof=1)
std_14 = StdTransform(in_column="target", seasonality=31, window=14, min_periods=14, out_column="std_14", ddof=1)
std_30 = StdTransform(in_column="target", seasonality=31, window=30, min_periods=30, out_column="std_30", ddof=1)


#cp_seg = ChangePointsSegmentationTransform(in_column="target")
fourier_31_7 = FourierTransform(in_column="target", period=31, order=7, out_column="fourier_31")
fourier_31_2 = FourierTransform(in_column="target", period=31, order=2, out_column="fourier_2")
fourier_7_2 = FourierTransform(in_column="target", period=31, order=2, out_column="fourier_7_2")
log_tr = LogTransform(in_column="target", out_column="target_log", inplace=False)  # создаем новую колонку target_log

date_flags = DateFlagsTransform(
    day_number_in_week=True,
    day_number_in_month=True,
    day_number_in_year=True,
    week_number_in_month=True,
    week_number_in_year=True,
    month_number_in_year=True,
    season_number=False,
    year_number=True,
    is_weekend=True,
    out_column="flag",
)
holiday_tr = HolidayTransform(out_column="holiday", iso_code="RUS")

transforms = [
    lags_small,
    lags,
    lags_macro,
    lags_macro_2,
    mean_tr_1,
    mean_tr_3,
    mean_tr_7,
    mean_tr_14,
    mean_tr_30,
    mean_tr_60,
    std_1,
    std_3,
    std_7,
    std_14,
    std_30,
    LogTransform(in_column="target"),  # создаем новую колонку target_log,
    holiday_tr,
    date_flags,
    #stl,
    #trend,
    imputer,
    fourier_31_7,
    fourier_31_2,
    #fourier_7_2,
    seg,
]
for transform in transforms:
    transform.fit(ts)
    ts = transform.transform(ts)

# Получаем итоговый DataFrame
transformed_df = ts.to_pandas(flatten=True).reset_index()

# Сохраняем в CSV
transformed_df.to_csv("tsdataset_with_lags.csv", index=False)
print("Сохранили TSDataset с лагами и трансформами в tsdataset_with_lags.csv")
print(df["target"].describe())

def objective(trial):
    model = CatBoostMultiSegmentModel(
        #iterations=trial.suggest_int("iterations", 300, 3000),
        #depth=trial.suggest_int("depth", 4, 16),
        #learning_rate=trial.suggest_float("learning_rate", 0.3, 0.9, log=True),
        #l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 0.01, 10.0, log=True),
        #random_strength=trial.suggest_float("random_strength", 0.01, 20.0, log=True),
        #bagging_temperature=trial.suggest_float('bagging_temperature', 0, 5.0),
        #border_count=trial.suggest_int('border_count', 32, 512),
        #one_hot_max_size=trial.suggest_int('one_hot_max_size', 2, 25),
        iterations = 873,
        depth = 10,
        learning_rate = 0.6112990073841404,
        l2_leaf_reg = 9.29277478806184,
        random_strength =0.07240417120087717,
        bagging_temperature = 1.5372496609872792,
        border_count = 41,
        one_hot_max_size = 21,
        loss_function='RMSE',
        logging_level='Silent',
        random_seed=42,
        early_stopping_rounds=200,
    )


#{'iterations': 873, 'depth': 10, 'learning_rate': 0.6112990073841404, 'l2_leaf_reg': 9.29277478806184,
# 'random_strength': 0.07240417120087717, 'bagging_temperature': 1.5372496609872792, 'border_count': 41,
# 'one_hot_max_size': 21}
#    with value: 243.85190461683024.

    pipeline = Pipeline(model=model, transforms=transforms, horizon=FORECAST_DAYS)

    pipeline.fit(train_ts)

    # Backtest на train_ts
    backtest_result = pipeline.backtest(
        ts=train_ts,
        metrics=[RMSE()],
        n_folds=7,
        mode="constant"
    )
    mean_rmse = backtest_result["metrics"]["RMSE"].mean().mean()
    return mean_rmse

# -----------------------------
# 4. Запускаем Optuna
# -----------------------------
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=1, show_progress_bar=True)  # можно увеличить n_trials для лучшего подбора

print("Best trial:", study.best_trial.params)

# -----------------------------
# 5. Финальная модель с лучшими параметрами
# -----------------------------
best_params = study.best_trial.params
final_model = CatBoostMultiSegmentModel(**best_params, logging_level="Silent")
pipeline = Pipeline(model=final_model, transforms=transforms, horizon=FORECAST_DAYS)
pipeline.fit(ts=train_ts)
forecast = pipeline.forecast(ts=train_ts, prediction_interval=True)

test_df = test_ts.to_pandas(flatten=True).reset_index()
forecast_df = forecast.to_pandas(flatten=True).reset_index()

test_df["timestamp"] = pd.to_datetime(test_df["timestamp"])
forecast_df["timestamp"] = pd.to_datetime(forecast_df["timestamp"])

# вытаскиваем article из segment (до "_")
test_df["article"] = test_df["segment"].str.split("|").str[0]
forecast_df["article"] = forecast_df["segment"].str.split("|").str[0]

# агрегируем по article + timestamp (сумма по департаментам)
test_agg = test_df.groupby(["timestamp", "article"], as_index=False)["target"].sum()
forecast_agg = forecast_df.groupby(["timestamp", "article"], as_index=False)["target"].sum()

# суммируем все артикулы (чтобы было 2 линии total)
total_fact = test_agg.groupby("timestamp", as_index=False)["target"].sum()
total_forecast = forecast_agg.groupby("timestamp", as_index=False)["target"].sum()

# =============================
# Рисуем факт vs прогноз
# =============================
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(
    total_fact["timestamp"], total_fact["target"],
    label="Actual (факт)", color="blue", linewidth=2
)
ax.plot(
    total_forecast["timestamp"], total_forecast["target"],
    label="Forecast (план)", color="red", linestyle="--", linewidth=2
)

ax.set_title("Forecast vs Actual (сумма по article)")
ax.set_xlabel("Date")
ax.set_ylabel("Target")
ax.legend()
plt.tight_layout()
plt.savefig(OUTPUT_PLOT, dpi=150)
plt.close()
print("Saved plot to", os.path.abspath(OUTPUT_PLOT))

# =============================
# Итоги по суммам
# =============================
fact_sum = total_fact["target"].sum()
forecast_sum = total_forecast["target"].sum()

print("=== Итоги (агрегация по article) ===")
print(f"Сумма факта за период: {fact_sum:.2f}")
print(f"Сумма прогноза за период: {forecast_sum:.2f}")
print(f"Разница (прогноз - факт): {forecast_sum - fact_sum:.2f}")