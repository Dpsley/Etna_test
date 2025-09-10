import pandas as pd
from etna.datasets import TSDataset
from etna.metrics import MAE, SMAPE, MSE, MAPE
from etna.analysis import plot_backtest
from etna.pipeline import Pipeline

METRICS = [MAE(), MSE(), MAPE(), SMAPE()]
DATA_CSV = "sales_remains_temp.csv"
df = pd.read_csv(DATA_CSV)
segment_name = 'АТ Москва|TALTHA-BP0026'

#print("=== Исходный DF ===")
#print(df.head())
#print(df.info())
#df.drop(["ProductName", "ProductType", "Article", "Department", "ProductProperties", "ProductGroup"], axis=1, inplace=True)
df = df[df['segment'] == segment_name].copy()

ts_df = TSDataset.to_dataset(df=df)
print("=== TSDataset.to_dataset ===")
print(ts_df.head())
print(ts_df.info())
ts = TSDataset(ts_df, freq="D")


from etna.transforms import (
    LinearTrendTransform,
    DensityOutliersTransform,
    TimeSeriesImputerTransform,
)

transforms = [
    # удаляем выбросы из данных
    DensityOutliersTransform(
        in_column="target",
        window_size=45,
        n_neighbors=25,
        distance_coef=1.9
    ),
    # заполняем образовавшиеся пропуски
    TimeSeriesImputerTransform(
        in_column="target",
        strategy="running_mean"
    ),
    # вычитаем тренд
    LinearTrendTransform(in_column="target"),
]

from etna.models import SeasonalMovingAverageModel, CatBoostMultiSegmentModel
from etna.pipeline import Pipeline

model = CatBoostMultiSegmentModel(seasonality=7, window=5)
pipeline = Pipeline(
    model=model,
    transforms=transforms,
    horizon=14
)


METRICS = [MAE(), MSE(), MAPE(), SMAPE()]

result = pipeline.backtest(
    ts=ts,
    metrics=METRICS,
    n_folds=5,
)

# result — это dict
print(result.keys())         # посмотри, какие есть ключи
metrics = result["metrics"]
forecasts = result["forecasts"]

plot_backtest(forecast_ts_list=forecasts, ts=ts, history_len=50)
metrics.head(7)
