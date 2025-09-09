import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from datetime import timedelta

# -------------------------
# Параметры
N_DAYS = 7  # на сколько дней прогноз
DATA_CSV = "data.csv"  # твой файл

# -------------------------
# Загружаем данные
df = pd.read_csv(DATA_CSV, parse_dates=['Date'])
df['Sold'] = df['Sold'].fillna(0)

# Словарь для хранения прогнозов
forecasts = {}

# Группируем по департаменту и артикулу
for (dept, article), group in df.groupby(['Department', 'Article']):
    # сортировка по дате
    ts = group.sort_values('Date').set_index('Date')['Sold']

    # агрегируем дубли на одну дату (суммируем продажи)
    ts = ts.groupby(ts.index).sum()

    # делаем временной ряд с ежедневной частотой
    ts = ts.asfreq('D', fill_value=0)

    # строим ARIMA (можно менять порядок)
    try:
        model = ARIMA(ts, order=(1, 1, 1))
        model_fit = model.fit()

        # прогноз на N_DAYS
        forecast = model_fit.forecast(N_DAYS)

        # сохраняем в словарь
        forecasts[(dept, article)] = forecast
    except Exception as e:
        print(f"Ошибка для {dept} | {article}: {e}")

# -------------------------
# Пример вывода прогноза для одного товара
for key, forecast in list(forecasts.items())[:3]:  # первые 3
    print(f"{key}:")
    print(forecast)
    print()
