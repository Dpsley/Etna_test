import pandas as pd

# Путь к твоему CSV
file_path = "sales_remains_072025.csv"

# Читаем CSV
df = pd.read_csv(file_path)

# Преобразуем колонку Date
df['timestamp'] = pd.to_datetime(df['Date'], format='%d.%m.%Y').dt.strftime('%Y-%m-%d')
df['segment'] = df['Department'] + '|' + df['Article']

df = df.rename(columns={"Date": "timestamp", "Sold": "target"})

# Записываем обратно в тот же файл
df.to_csv(file_path, index=False)

print("Готово, даты преобразованы.")
