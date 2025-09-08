import pandas as pd

# Путь к твоему CSV
file_path = "sales_remains_temp.csv"

# Читаем CSV
df = pd.read_csv(file_path)

# Преобразуем колонку Date
df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d.%m.%Y').dt.strftime('%Y-%m-%d')
df['segment'] = df['Department'] + '|' + df['Article']


# Записываем обратно в тот же файл
df.to_csv(file_path, index=False)

print("Готово, даты преобразованы.")
