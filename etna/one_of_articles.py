import pandas as pd
import matplotlib.pyplot as plt
from etna.auto import Auto
from etna.datasets import TSDataset
from etna.metrics import SMAPE, MAPE, MAE, RMSE
from etna.analysis import plot_backtest

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

DATA_CSV = "expanded_etna.csv"
FORECAST_DAYS = 7
SEGMENT_FILTER = "АТ Москва|TALTHA-BP0026"
OUTPUT_PLOT = "auto_backtest.png"

# -----------------------------
# 1. Загружаем данные
# -----------------------------
df = pd.read_csv(DATA_CSV)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values(['segment', 'timestamp']).reset_index(drop=True)

# фильтруем только нужный сегмент
df_filtered = df[df['segment'] == SEGMENT_FILTER].copy()

# -----------------------------
# 2. Создаём TSDataset
# -----------------------------
ts = TSDataset(df_filtered, freq="D")
print(ts)
# -----------------------------
# 3. Настраиваем AutoModel
# -----------------------------
auto_model=Auto(
#    metrics=[SMAPE(), MAPE(), MAE(), RMSE()],
    target_metric=SMAPE(),  # метрика для оптимизации
    horizon=7,  # горизонт прогноза
    #metric_aggregation='mean',  # метод агрегации по сегментам
    experiment_folder='auto_experiment',  # папка для сохранения результатов
    #runner=None,  # использование стандартного локального раннера
    #storage=None,
    # использование стандартного хранилища Optuna
    )

# -----------------------------
# 4. Обучаем
# -----------------------------
auto_model.fit(ts, catch=(Exception,))

# -----------------------------
# 5. Делам backtest
# -----------------------------
backtest_result = auto_model.backtest(ts, n_folds=3)

print(backtest_result)

# Get all metrics of greedy search
#print(auto_model.summary())