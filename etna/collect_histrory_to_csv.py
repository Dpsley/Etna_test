import pandas as pd

# Загружаем файлы
df_hist = pd.read_csv("sales_remains_072023_062025.csv")
df_forecast = pd.read_csv("sales_remains_072025.csv")

# Функция подготовки датафрейма: timestamp -> год-месяц, группировка по Department + Article
def prepare_df(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['year_month'] = df['timestamp'].dt.to_period('M').astype(str)
    # Агрегация по департаменту, артикулу и месяцу
    df_agg = df.groupby(['Department', 'Article', 'year_month'])['target'].sum().reset_index()
    return df_agg

df_hist_agg = prepare_df(df_hist)
df_forecast_agg = prepare_df(df_forecast)

# Объединяем исторические и прогнозные данные
df_all = pd.concat([df_hist_agg, df_forecast_agg], ignore_index=True)

# Пивотируем: строки = Department+Article, колонки = year_month, значения = target
df_pivot = df_all.pivot_table(index=['Department', 'Article'],
                              columns='year_month',
                              values='target',
                              fill_value=0).reset_index()

# Сортировка колонок по дате
cols_sorted = ['Department', 'Article'] + sorted([col for col in df_pivot.columns if col not in ['Department', 'Article']])
df_pivot = df_pivot[cols_sorted]

# Сохраняем в Excel
df_pivot.to_csv("sales_by_month.csv", index=False)

print("Готово! Файл sales_by_month.xlsx создан.")

8567.694386
1202.888598
1300.360475
101.042573
129.445596
136.334634
8033.971804
854.236932
906.615902
866.116446
4652.492659
5890.938737
290.412340
101.042573
101.042573
899.990690
101.042573
101.042573

