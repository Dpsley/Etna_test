import pandas as pd

# Путь к твоему CSV
file_path = "qu_TALTHA-DP0082.csv"

# Читаем CSV
df = pd.read_csv(file_path)

# Преобразуем колонку Date
#df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
df['segment'] = df["Client"] + '|' + df['Department'] + '|' + df['Article']
#df.drop(["Unnamed: 10"], axis=1, inplace=True)
df = df.rename(columns={"Date": "timestamp", "Sold": "target"})

# Записываем обратно в тот же файл
df.to_csv(file_path, index=False)

print("Готово, даты преобразованы.")
