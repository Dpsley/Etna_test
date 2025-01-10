import pandas as pd
from etna.models import CatBoostMultiSegmentModel
from etna.datasets import TSDataset
from etna.pipeline import Pipeline
from etna.transforms import LagTransform, MeanTransform
from etna.metrics import MAE
import matplotlib.pyplot as plt

# Чтение данных
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

# Создаем временной ряд для работы с etna
df_grouped = df_grouped.rename(columns={'Номенклатура / Сеть': 'segment'})

# Преобразуем DataFrame в формат TSDataset
df_grouped['timestamp'] = pd.to_datetime(df_grouped['Дата'])
df_grouped = df_grouped.drop(columns=['Дата'])  # Удалим старый столбец

# Преобразуем в TSDataset
df_grouped = TSDataset.to_dataset(df_grouped)

# Создаем TSDataset
ts = TSDataset(df_grouped, freq='MS')

# Проверяем структуру TSDataset
print(ts.df.head())

# Убедитесь, что колонка 'target' существует
# Создаем колонку 'target' из 'Продажи'
# Доступ к колонке 'Продажи' из MultiIndex
print("\n")

print(ts.df.columns)
print(ts.df.index)

segments = ts.df.index.get_level_values('segment').unique()

# Создаем список продаж для каждого сегмента
sales_column = [(segment, 'Продажи') for segment in segments]

# Проверяем результат
print(sales_column)

# Теперь создаём целевой столбец 'target' для прогноза
ts.df['target'] = ts.df.xs('Продажи', level='feature', axis=1)
print(ts.df.head())

# Инициализация и обучение модели
# Используйте правильные обращения к данным внутри DataFrame

# Создаем список трансформаций
transforms = [
    LagTransform(in_column='target', lags=[1]),  # Добавляем лаг (задержку) с периодом 1
    MeanTransform(in_column='target', window=3)  # Добавляем скользящее среднее с окном 3
]

# Применяем преобразования
ts.fit_transform(transforms)

# Разделяем данные на тренировочную и тестовую выборки
train, test = TSDataset.train_test_split(ts, test_size=12)

# Инициализируем и обучаем модель
model = CatBoostMultiSegmentModel(iterations=1000, learning_rate=0.05, depth=6, silent=True)
pipeline = Pipeline(model=model, transforms=transforms, horizon=12)

# Обучаем
pipeline.fit(train)

# Прогнозирование
forecast = pipeline.forecast()

# Визуализация результата
for segment in forecast.segments:
    plt.figure(figsize=(10, 6))
    plt.plot(train[:, segment, "target"].index, train[:, segment, "target"].values, label='Исторические данные')
    plt.plot(test[:, segment, "target"].index, test[:, segment, "target"].values, label='Тестовые данные')
    plt.plot(forecast[:, segment, "target"].index, forecast[:, segment, "target"].values, label='Прогноз')
    plt.title(f'Прогноз продаж для {segment}')
    plt.xlabel('Дата')
    plt.ylabel('Продажи')
    plt.legend()
    plt.show()

# Оцениваем точность прогноза
metric = MAE()
mae_value = metric(y_true=test, y_pred=forecast)
print(f'MAE: {mae_value}')

# Экспортируем прогнозы в CSV
forecast.to_pandas(flatten=True).to_csv('forecast_results_catboost.csv', index=False, encoding='utf-8')
