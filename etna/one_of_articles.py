import pandas as pd
from etna.datasets import TSDataset
from etna.pipeline import Pipeline
from etna.models import CatBoostMultiSegmentModel
from etna.transforms import LagTransform, DateFlagsTransform, SegmentEncoderTransform
from etna.metrics import SMAPE
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import optuna

# -----------------------------
# Настройки
# -----------------------------
DATA_CSV = "sales_remains_temp.csv"
FORECAST_DAYS = 30
OUTPUT_PLOT = "auto_backtest.png"

os.environ["CUDA_VISIBLE_DEVICES"] = ""

# -----------------------------
# 1. Загружаем данные
# -----------------------------
df = pd.read_csv(DATA_CSV)
print("=== Исходный DF ===")
print(df.head())
print(df.info())

# Если нужно — добавляем колонку сегмента вручную, чтобы видеть как формируется TSDataset
if 'segment' not in df.columns:
    df['segment'] = df['article'] + "_" + df['department']  # пример
    print("=== DF с сегментами ===")
    print(df[['Date', 'segment']].head())
ts_df = TSDataset.to_dataset(df=df)
print("=== TSDataset.to_dataset ===")
print(ts_df.head())
print(ts_df.info())
ts = TSDataset(ts_df, freq="D")
train_ts, test_ts = ts.train_test_split(test_size=FORECAST_DAYS)
print("=== Сегменты TSDataset ===")
print(ts.segments)
print("=== Train/ Test shapes ===")
print(train_ts[:, :, "target"].shape, test_ts[:, :, "target"].shape)
# -----------------------------
# 2. Трансформеры
# -----------------------------
seg = SegmentEncoderTransform()
lags = LagTransform(in_column="target", lags=list(range(FORECAST_DAYS, 60 + FORECAST_DAYS)), out_column="lag")
date_flags = DateFlagsTransform(
    day_number_in_week=True,
    day_number_in_month=True,
    day_number_in_year=True,
    week_number_in_month=True,
    week_number_in_year=True,
    month_number_in_year=True,
    year_number=True,
    is_weekend=True,
    out_column="flag",
)
transforms = [lags, date_flags, seg]


# -----------------------------
# 3. Определяем функцию для Optuna
# -----------------------------
def objective(trial):
    model = CatBoostMultiSegmentModel(
        iterations=trial.suggest_int("iterations", 500, 2000),
        depth=trial.suggest_int("depth", 4, 10),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.9, log=True),
        l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 0.01, 10.0, log=True),
        random_strength=trial.suggest_float("random_strength", 0.01, 20.0, log=True),
        bagging_temperature=trial.suggest_float('bagging_temperature', 0, 5.0),
        border_count=trial.suggest_int('border_count', 32, 256),
        one_hot_max_size=trial.suggest_int('one_hot_max_size', 2, 25),
        loss_function='RMSE',
        logging_level='Silent',
        random_seed=42,
        early_stopping_rounds=200,
    )

    pipeline = Pipeline(model=model, transforms=transforms, horizon=FORECAST_DAYS)
    pipeline.fit(ts=train_ts)

    forecast = pipeline.forecast(ts=train_ts, prediction_interval=False, n_folds=1)

    # тут исправление: берем средний SMAPE по всем сегментам
    smape_scores = SMAPE()(test_ts, forecast)
    if isinstance(smape_scores, dict):
        smape_score = sum(smape_scores.values()) / len(smape_scores)
    else:
        smape_score = smape_scores

    return smape_score



# -----------------------------
# 4. Запускаем Optuna
# -----------------------------
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100, show_progress_bar=True)  # можно увеличить n_trials для лучшего подбора

print("Best trial:", study.best_trial.params)

# -----------------------------
# 5. Финальная модель с лучшими параметрами
# -----------------------------
best_params = study.best_trial.params
final_model = CatBoostMultiSegmentModel(**best_params, logging_level="Info")
pipeline = Pipeline(model=final_model, transforms=transforms, horizon=FORECAST_DAYS)
pipeline.fit(ts=train_ts)
forecast = pipeline.forecast(ts=train_ts, prediction_interval=True)

fig, ax = plt.subplots(figsize=(12, 6))

for segment in test_ts.segments:
    ax.plot(
        test_ts[:, segment, "target"],
        label=f"Actual {segment}",
        color="blue"
    )
    ax.plot(
        forecast[:, segment, "target"],
        label=f"Forecast {segment}",
        color="red",
        linestyle="--"
    )

ax.set_title("Forecast vs Actual")
ax.set_xlabel("Date")
ax.set_ylabel("Target")
ax.legend()
plt.tight_layout()
plt.savefig(OUTPUT_PLOT, dpi=150)
plt.close()
print("Saved plot to", os.path.abspath(OUTPUT_PLOT))