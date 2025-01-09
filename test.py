import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.stattools import adfuller


# Примерный формат данных, которые ты хочешь прогнозировать
# Сначала очищаем и преобразуем данные
df = pd.read_csv('sales2.csv')

# Переводим месяцы на английский для корректного парсинга
month_translation = {
    'Январь': 'January', 'Февраль': 'February', 'Март': 'March', 'Апрель': 'April',
    'Май': 'May', 'Июнь': 'June', 'Июль': 'July', 'Август': 'August', 'Сентябрь': 'September',
    'Октябрь': 'October', 'Ноябрь': 'November', 'Декабрь': 'December'
}


def translate_month(month):
    for ru_month, en_month in month_translation.items():
        if month.startswith(ru_month):
            return month.replace(ru_month, en_month)
    return month


# Преобразуем столбцы с месяцами
df.columns = [translate_month(col) if isinstance(col, str) else col for col in df.columns]

# Преобразуем данные в формат временного ряда
df_melted = df.melt(id_vars=["Код товара", "Номенклатура / Сеть", "Группа", "Марка", "Тип товара"],
                    var_name="Дата", value_name="Продажи")

# Преобразуем колонку "Дата" в datetime
df_melted['Дата'] = pd.to_datetime(df_melted['Дата'], format='%B %Y')

# Преобразуем значения "Продажи" в числовой формат
df_melted['Продажи'] = pd.to_numeric(df_melted['Продажи'], errors='coerce')

# Группируем данные по месяцам и товарам
df_grouped = df_melted.groupby(['Дата', 'Номенклатура / Сеть'])['Продажи'].sum().reset_index()
df_grouped['Продажи'] = df_grouped['Продажи'].rolling(window=3, min_periods=1).mean()

# Создаем DataFrame для записи прогнозов
final_results = pd.DataFrame()

# Для каждого товара строим отдельный прогноз
for product in df_grouped['Номенклатура / Сеть'].unique():
    # Фильтруем данные для конкретного товара
    product_data = df_grouped[df_grouped['Номенклатура / Сеть'] == product]

    # Подготовка данных для ARIMA
    product_data = product_data[['Дата', 'Продажи']].rename(columns={'Дата': 'ds', 'Продажи': 'y'})
    product_data['y'] = np.log(product_data['y'].replace(0, np.nan))  # если есть нули, заменим их на NaN
    product_data = product_data.dropna()  # удаляем строки с NaN

    product_data.set_index('ds', inplace=True)

    # Устанавливаем частоту данных (ежемесячно)
    product_data = product_data.asfreq('MS')

    # Инициализируем модель ARIMA (p, d, q) - подбираем параметры
    model = ARIMA(product_data['y'], order=(9, 1, 1))  # p=5, d=1, q=0 как пример, параметры можно настроить
    model_fit = model.fit()

    # Прогнозируем 12 месяцев вперед
    forecast = model_fit.forecast(steps=12)

    # Визуализируем прогноз
    plt.figure(figsize=(10, 6))
    plt.plot(product_data.index, product_data['y'], label='Исторические данные')
    plt.plot(pd.date_range(product_data.index[-1], periods=13, freq='M')[1:], forecast, label='Прогноз')
    plt.title(f'Прогноз продаж товара: {product}')
    plt.xlabel('Месяц')
    plt.ylabel('Продажи')
    plt.legend()
    plt.show()
    result = adfuller(product_data['y'])
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    # Добавляем прогноз в финальные результаты
    forecast_df = pd.DataFrame(forecast, columns=['yhat'],
                               index=pd.date_range(product_data.index[-1], periods=13, freq='M')[1:])
    forecast_df['Номенклатура / Сеть'] = product

    # Округляем значения
    forecast_df['yhat'] = forecast_df['yhat'].astype(float).round()

    # Собираем данные для финальной таблицы
    final_results = pd.concat([final_results, forecast_df[['Номенклатура / Сеть', 'yhat']]], axis=0)

# Пивотируем таблицу, чтобы каждая дата была отдельной колонкой для всех товаров
final_results_pivoted = final_results.pivot(index='Номенклатура / Сеть', columns=final_results.index, values='yhat')

# Записываем финальные результаты в CSV
final_results_pivoted.to_csv('forecast_results_arima.csv', index=True, encoding='utf-8')

# Проверяем содержимое файла
print(final_results_pivoted.head())
