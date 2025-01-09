import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error


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

    # Подготовка данных для Prophet
    product_data = product_data[['Дата', 'Продажи']].rename(columns={'Дата': 'ds', 'Продажи': 'y'})

    # Инициализируем модель Prophet
    model = Prophet(yearly_seasonality=False, changepoint_prior_scale=0.05)
    #model.add_seasonality(name='quarterly', period=3, fourier_order=3)
    model.add_seasonality(name='quarterly', period=3, fourier_order=3)
    model.add_seasonality(name='monthly', period=1, fourier_order=5)

    # Обучаем модель на данных
    model.fit(product_data)

    # Создаем будущие даты для прогноза (12 месяцев вперед)
    future = model.make_future_dataframe(periods=12, freq='M')

    # Прогнозируем
    forecast = model.predict(future)

    # Визуализируем прогноз
    fig = model.plot(forecast)
    plt.title(f'Прогноз продаж товара: {product}')
    plt.xlabel('Месяц')
    plt.ylabel('Продажи')
    plt.show()

    # Добавляем только необходимые данные для финального результата
    forecast['Номенклатура / Сеть'] = product
    forecast['yhat'] = forecast['yhat'].round()
    forecast['yhat_lower'] = forecast['yhat_lower'].round()
    forecast['yhat_upper'] = forecast['yhat_upper'].round()

    # Собираем данные для финальной таблицы
    forecast = forecast[['Номенклатура / Сеть', 'ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    # Пивотируем таблицу, чтобы каждая дата была отдельной колонкой для всех товаров
    forecast_pivoted = forecast.pivot(index='Номенклатура / Сеть', columns='ds', values='yhat')
    forecast_pivoted_lower = forecast.pivot(index='Номенклатура / Сеть', columns='ds', values='yhat_lower')
    forecast_pivoted_upper = forecast.pivot(index='Номенклатура / Сеть', columns='ds', values='yhat_upper')

    # Объединяем все колонки в одну таблицу
    result = pd.concat([forecast_pivoted, forecast_pivoted_lower, forecast_pivoted_upper], axis=1)

    # Добавляем название товара как индекс
    result.columns = [f'Продажи_{col.strftime("%Y-%m")}' for col in result.columns[:12]] + \
                     [f'Нижняя_граница_{col.strftime("%Y-%m")}' for col in result.columns[12:24]] + \
                     [f'Верхняя_граница_{col.strftime("%Y-%m")}' for col in result.columns[24:]]

    # Добавляем в финальный результат
    final_results = pd.concat([final_results, result], axis=0)

# Записываем финальные результаты в CSV
final_results.to_csv('forecast_results.csv', index=True, encoding='utf-8')

# Проверяем содержимое файла
print(final_results.head())
