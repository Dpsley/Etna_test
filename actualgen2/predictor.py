import os
import pandas as pd
import argparse
from datetime import timedelta

# -------------------------------
# Парсинг аргументов
# -------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--department", required=True, help="Название департамента")
parser.add_argument("--article", required=True, help="Артикул товара")
parser.add_argument("--days", type=int, required=True, help="Количество дней для прогноза")
args = parser.parse_args()

# -------------------------------
# Загружаем данные
# -------------------------------
df = pd.read_csv('expanded.csv', sep=';', parse_dates=['Date'], dayfirst=False)

# Отсечка дат: только до 30 июня
cutoff_date = pd.Timestamp(year=df['Date'].dt.year.min(), month=6, day=30)
df = df[df['Date'] <= cutoff_date]

# Фильтруем по департаменту и артикулу
filtered = df[(df['Department'] == args.department) & (df['Article'] == args.article)].copy()

if filtered.empty:
    print("Записи не найдены")
    exit()

filtered = filtered.sort_values('Date')
last_date = filtered['Date'].iloc[-1]

# -------------------------------
# Создаем будущие даты с Sold = 0
# -------------------------------
future_dates = [last_date + timedelta(days=i) for i in range(1, args.days + 1)]
future_df = pd.DataFrame({'Date': future_dates})
future_df['Sold'] = 0  # пока 0, чтобы считать лаги

# Добавляем все колонки товара, кроме лагов, и заполняем значениями последней строки
product_cols = [c for c in df.columns if c not in ['Date','Sold','sold_lag1','sold_lag7','sold_lag14','sold_lag28','sold_ma7','sold_ma14']]
for col in product_cols:
    future_df[col] = filtered.iloc[-1][col]

# -------------------------------
# Соединяем историю и прогноз, чтобы посчитать лаги
# -------------------------------
combined = pd.concat([filtered, future_df], ignore_index=True).sort_values('Date')

# Функция для вычисления лагов
def calc_lags(s):
    s_lag1 = s.shift(1).fillna(0)
    s_lag7 = s.shift(7).fillna(0)
    s_lag14 = s.shift(14).fillna(0)
    s_lag28 = s.shift(28).fillna(0)
    s_ma7 = s.shift(1).rolling(7, min_periods=1).mean()
    s_ma14 = s.shift(1).rolling(14, min_periods=1).mean()
    s_ma28 = s.rolling(28, min_periods=1).mean()
    return pd.DataFrame({
        'sold_lag1': s_lag1,
        'sold_lag7': s_lag7,
        'sold_lag14': s_lag14,
        'sold_lag28': s_lag28,
        'sold_ma7': s_ma7,
        'sold_ma14': s_ma14
    })

lags_df = calc_lags(combined['Sold'])
combined = pd.concat([combined.drop(columns=['sold_lag1','sold_lag7','sold_lag14','sold_lag28','sold_ma7','sold_ma14'], errors='ignore'), lags_df], axis=1)

# -------------------------------
# Берем только прогнозные даты
# -------------------------------
forecast_df = combined[combined['Date'].isin(future_dates)].reset_index(drop=True)

# -------------------------------
# Сохраняем
# -------------------------------
out_path = os.path.join(os.getcwd(), 'forecast_only.csv')
forecast_df.to_csv(out_path, sep=';', index=False)
print(f"Прогноз сохранен в {out_path}")
print(forecast_df)
