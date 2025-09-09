import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Загружаем прогнозные данные (july.csv)
df_pred = pd.read_csv("july.csv", parse_dates=['Date'], dayfirst=True)
df_pred = df_pred.rename(columns={'Sold': 'Forecast'})  # прогноз

# Загружаем фактические данные (sales_remains_072025.csv)
df_fact = pd.read_csv("sales_remains_072025.csv", parse_dates=['Date'], dayfirst=True)
# Исправляем формат даты, если надо
df_fact['Date'] = pd.to_datetime(df_fact['Date'], format='%d.%m.%Y').dt.strftime('%Y-%m-%d')
df_fact = df_fact.rename(columns={'Sold': 'Actual'})  # факт

# Мержим факт и прогноз
df = pd.merge(df_fact, df_pred, on=['Department', 'Article', 'Date'], how='left')

# Фильтруем июнь-июль 2025
df = df[(df['Date'] >= '2025-06-01') & (df['Date'] <= '2025-07-31')]

# Создаём папку для графиков
os.makedirs("plots", exist_ok=True)

# Рисуем по каждому артикулу
for dept in df['Department'].unique():
    dept_df = df[df['Department'] == dept]
    for art in dept_df['Article'].unique():
        art_df = dept_df[dept_df['Article'] == art].sort_values('Date')

        # Суммы факта и прогноза
        total_actual = art_df['Actual'].sum()
        total_forecast = art_df['Forecast'].sum()
        print(f"{dept} - {art} ({art_df['Article'].iloc[0]}): "
              f"Факт={total_actual}, Прогноз={total_forecast}")

        plt.figure(figsize=(12, 6))
        sns.lineplot(x='Date', y='Actual', data=art_df, label='Факт', marker='o')
        if 'Forecast' in art_df and art_df['Forecast'].notna().any():
            sns.lineplot(x='Date', y='Forecast', data=art_df, label='Прогноз', marker='x')

        plt.title(f'{dept} - {art} ({art_df["Article"].iloc[0]})')
        plt.xlabel('Дата')
        plt.ylabel('Продажи')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        filename = f"plots/{dept}_{art}.png".replace(" ", "_")
        plt.savefig(filename)
        plt.close()
