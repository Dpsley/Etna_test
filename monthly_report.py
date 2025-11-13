import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta


def create_combined_report(csv_file_path, json_file_path, output_file_path=None):
    """
    Создает объединенный отчет с историческими данными, прогнозами и итоговыми колонками за 123 дня
    """

    # Загрузка исторических данных из CSV
    print("Загрузка исторических данных...")
    df = pd.read_csv(csv_file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['month'] = df['timestamp'].dt.strftime('%Y-%m')
    df['Article'] = df['segment'].str.split('|').str[-1]

    # Загрузка прогнозов из JSON
    print("Загрузка прогнозных данных...")
    with open(json_file_path, 'r', encoding='utf-8') as f:
        forecast_data = json.load(f)

    # Функция для обработки чисел: обрезка отрицательных и округление
    def process_number(x):
        return max(0, round(x))

    # Расчет last_123_days_sum для каждого товара
    print("Расчет суммы за последние 123 дня...")
    latest_date = df['timestamp'].max()
    cutoff_date = latest_date - timedelta(days=123)

    last_123_days = df[df['timestamp'] >= cutoff_date]
    last_123_sum = last_123_days.groupby(['Article'])['target'].sum().reset_index()
    last_123_sum['last_3_month_sum'] = last_123_sum['target'].apply(process_number)
    last_123_sum = last_123_sum[['Article', 'last_3_month_sum']]

    # Расчет forecast сумм за 123 дня для каждого товара
    print("Расчет прогнозных сумм за 123 дня...")
    forecast_rows = []
    for segment, forecasts in forecast_data.items():
        article = segment.split('|')[-1] if '|' in segment else segment

        # Суммируем все значения для этого товара
        total_mean = sum(forecast['target'] for forecast in forecasts)
        total_lower = sum(forecast['target_lower'] for forecast in forecasts)
        total_upper = sum(forecast['target_upper'] for forecast in forecasts)

        forecast_rows.append({
            'Article': article,
            'forecast_3_month_mean': process_number(total_mean),
            'forecast_3_month_lower': process_number(total_lower),
            'forecast_3_month_upper': process_number(total_upper)
        })

    forecast_sums = pd.DataFrame(forecast_rows)

    # Преобразование JSON данных в DataFrame для основной таблицы
    forecast_rows_detailed = []
    for segment, forecasts in forecast_data.items():
        article = segment.split('|')[-1] if '|' in segment else segment

        for forecast in forecasts:
            forecast_rows_detailed.append({
                'Article': article,
                'timestamp': forecast['timestamp'],
                'target_mean': forecast['target'],
                'target_lower': forecast['target_lower'],
                'target_upper': forecast['target_upper']
            })

    forecast_df = pd.DataFrame(forecast_rows_detailed)
    forecast_df['timestamp'] = pd.to_datetime(forecast_df['timestamp'])
    forecast_df['month'] = forecast_df['timestamp'].dt.strftime('%Y-%m')

    print(f"Загружено прогнозов: {len(forecast_df)} записей")
    print(f"Уникальных товаров в прогнозах: {forecast_df['Article'].nunique()}")

    # Агрегация исторических данных по месяцам
    print("Агрегация исторических данных...")
    historical_monthly = df.groupby(['Article', 'month'])['target'].sum().reset_index()
    historical_monthly['target'] = historical_monthly['target'].apply(process_number)

    # Агрегация прогнозных данных по месяцам
    print("Агрегация прогнозных данных...")
    forecast_monthly = forecast_df.groupby(['Article', 'month']).agg({
        'target_mean': 'sum',
        'target_lower': 'sum',
        'target_upper': 'sum'
    }).reset_index()

    # Обрабатываем прогнозные данные
    forecast_monthly['target_mean'] = forecast_monthly['target_mean'].apply(process_number)
    forecast_monthly['target_lower'] = forecast_monthly['target_lower'].apply(process_number)
    forecast_monthly['target_upper'] = forecast_monthly['target_upper'].apply(process_number)

    # Создание сводных таблиц для исторических данных
    historical_pivot = historical_monthly.pivot_table(
        index=['Article'],
        columns='month',
        values='target',
        aggfunc='sum',
        fill_value=0
    ).reset_index()

    # Создание сводных таблиц для прогнозов (3 отдельные таблицы)
    forecast_mean_pivot = forecast_monthly.pivot_table(
        index=['Article'],
        columns='month',
        values='target_mean',
        aggfunc='sum',
        fill_value=0
    ).reset_index()

    forecast_lower_pivot = forecast_monthly.pivot_table(
        index=['Article'],
        columns='month',
        values='target_lower',
        aggfunc='sum',
        fill_value=0
    ).reset_index()

    forecast_upper_pivot = forecast_monthly.pivot_table(
        index=['Article'],
        columns='month',
        values='target_upper',
        aggfunc='sum',
        fill_value=0
    ).reset_index()

    # Переименование колонок для прогнозов
    def rename_forecast_columns(df, suffix):
        new_columns = []
        for col in df.columns:
            if col in ['Article']:
                new_columns.append(col)
            else:
                new_columns.append(f"{col}-forecast_{suffix}")
        df.columns = new_columns
        return df

    forecast_mean_pivot = rename_forecast_columns(forecast_mean_pivot, 'mean')
    forecast_lower_pivot = rename_forecast_columns(forecast_lower_pivot, 'lower')
    forecast_upper_pivot = rename_forecast_columns(forecast_upper_pivot, 'upper')

    # Объединение всех данных
    print("Объединение данных...")

    # Начинаем с исторических данных
    combined_df = historical_pivot.rename(columns={
        'Article': 'Артикул'
    })

    # Объединяем с прогнозами
    combined_df = combined_df.merge(
        forecast_mean_pivot.rename(columns={'Article': 'Артикул'}),
        on=['Артикул'],
        how='outer'
    )

    combined_df = combined_df.merge(
        forecast_lower_pivot.rename(columns={'Article': 'Артикул'}),
        on=['Артикул'],
        how='outer'
    )

    combined_df = combined_df.merge(
        forecast_upper_pivot.rename(columns={'Article': 'Артикул'}),
        on=['Артикул'],
        how='outer'
    )

    # Добавляем колонку last_123_days_sum
    combined_df = combined_df.merge(
        last_123_sum.rename(columns={'Article': 'Артикул'}),
        on=['Артикул'],
        how='left'
    )

    # Добавляем прогнозные колонки за 123 дня
    combined_df = combined_df.merge(
        forecast_sums.rename(columns={'Article': 'Артикул'}),
        on=['Артикул'],
        how='left'
    )

    # Заполняем NaN значения нулями
    combined_df = combined_df.fillna(0)

    # Обрабатываем все числовые колонки в финальном DataFrame (на всякий случай)
    print("Финальная обработка числовых данных...")
    numeric_columns = [col for col in combined_df.columns if col not in ['Артикул']]
    for col in numeric_columns:
        combined_df[col] = combined_df[col].apply(process_number)

    # Сортировка колонок - итоговые колонки в конец
    all_columns = combined_df.columns.tolist()
    dept_article_cols = ['Артикул']
    summary_cols = ['last_3_month_sum', 'forecast_3_month_mean', 'forecast_3_month_lower', 'forecast_3_month_upper']

    other_columns = [col for col in all_columns if col not in dept_article_cols + summary_cols]

    # Разделяем колонки на исторические и прогнозные
    historical_cols = [col for col in other_columns if 'forecast' not in col]
    forecast_cols = [col for col in other_columns if 'forecast' in col]

    # Сортируем отдельно исторические и прогнозные колонки
    historical_cols_sorted = sorted(historical_cols)
    forecast_cols_sorted = sorted(forecast_cols)

    # Комбинируем в правильном порядке: исторические -> прогнозные -> итоговые
    final_columns = dept_article_cols + historical_cols_sorted + forecast_cols_sorted + summary_cols
    combined_df = combined_df[final_columns]

    # Сохранение результатов
    if output_file_path:
        combined_df.to_csv(output_file_path, index=False, encoding='utf-8-sig')
        print(f"Объединенный отчет сохранен в: {output_file_path}")

    # Статистика
    print(f"\nСтатистика отчета:")
    print(f"Уникальных артикулов: {combined_df['Артикул'].nunique()}")
    print(f"Исторических месяцев: {len(historical_cols_sorted)}")
    print(f"Прогнозных месяцев: {len(forecast_cols_sorted) // 3}")
    print(
        f"Итоговые колонки: last_3_month_sum, forecast_3_month_mean, forecast_3_month_lower, forecast_3_month_upper")

    return combined_df


def print_sample_results(df, num_rows=5):
    """Выводит sample результатов"""
    print("\n" + "=" * 120)
    print("ПРЕВЬЮ РЕЗУЛЬТАТОВ (первые {} строк):".format(num_rows))
    print("=" * 120)

    # Показываем первые несколько колонок и итоговые колонки
    first_columns = df.columns[:4].tolist()  # Первые 4 колонки
    last_columns = df.columns[-4:].tolist()  # Последние 4 колонки (итоговые)

    display_columns = first_columns + last_columns
    print(df[display_columns].head(num_rows).to_string(index=False))


def main():
    # Укажите пути к файлам
    csv_file = "output_monthly.csv"
    json_file = "forecast.json"
    output_file = "combined_sales_forecast_report_monthly.csv"

    try:
        # Создаем объединенный отчет
        results = create_combined_report(csv_file, json_file, output_file)

        # Выводим sample результатов
        print_sample_results(results, 8)

        # Дополнительная информация
        print(f"\nПолные результаты содержат:")
        print(f"- Строк: {len(results)}")
        print(f"- Колонок: {len(results.columns)}")
        print(f"- Все числа обрезаны от отрицательных значений и округлены до целых")
        print(f"- Добавлены 4 итоговые колонки за 123 дня")

    except FileNotFoundError as e:
        print(f"Файл не найден: {e}")
    except json.JSONDecodeError as e:
        print(f"Ошибка чтения JSON файла: {e}")
    except Exception as e:
        print(f"Произошла ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
