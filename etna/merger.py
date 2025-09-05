import pandas as pd

# Загружаем данные
df_etna = pd.read_csv('expanded_etna.csv')
df_sales = pd.read_csv('sales_remains.csv')

# Преобразуем дату в sales_remains.csv: '01.01.2025' -> '2025-01-01'
df_sales['Date'] = pd.to_datetime(df_sales['Date'], format='%d.%m.%Y').dt.strftime('%Y-%m-%d')

# Переименовываем для совпадения с expanded_etna
df_sales.rename(columns={'Date': 'timestamp', 'Sold': 'Sold_temp'}, inplace=True)

# Оставляем только нужные столбцы: сопоставление по Department, Article, timestamp
df_sales_subset = df_sales[['Department', 'Article', 'timestamp', 'Sold_temp']]

# Преобразуем timestamp в expanded_etna к строке (на всякий случай)
df_etna['timestamp'] = pd.to_datetime(df_etna['timestamp']).dt.strftime('%Y-%m-%d')

# Слияние по Department, Article, timestamp
df_merged = pd.merge(
    df_etna,
    df_sales_subset,
    on=['Department', 'Article', 'timestamp'],
    how='left'  # чтобы сохранить все строки из etna
)

# Обновляем target: если есть Sold_temp — используем его, иначе оставляем старый target
df_merged['target'] = df_merged['Sold_temp'].fillna(df_merged['target']).astype(int)

# Убираем временный столбец
df_merged.drop(columns=['Sold_temp'], inplace=True)

# Проверка: пересоздаём segment (на всякий случай, чтобы был корректным)
df_merged['segment'] = df_merged['Department'] + '|' + df_merged['Article']

# Сохраняем обратно в expanded_etna.csv
df_merged.to_csv('expanded_etna.csv', index=False)

print("Файл expanded_etna.csv успешно обновлён: target обновлён из sales_remains.csv, segment перегенерирован.")