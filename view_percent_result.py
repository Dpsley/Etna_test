import pandas as pd


def calculate_accuracy_metrics(csv_file):
    # Чтение CSV файла
    df = pd.read_csv(csv_file)

    # Проверка наличия необходимых колонок
    required_columns = ['last_123_days_sum', 'forecast_123_days_mean']
    if not all(col in df.columns for col in required_columns):
        print("Ошибка: отсутствуют необходимые колонки в CSV файле")
        return

    # Функция для расчета процента попадания (ИСПРАВЛЕННАЯ)
    def calculate_accuracy(row):
        actual = row['last_123_days_sum']
        forecast = row['forecast_123_days_mean']

        # Если оба значения равны 0 - исключаем из расчета или считаем отдельно
        if actual == 0 and forecast == 0:
            return None  # Исключаем такие случаи из расчета

        # Если одно из значений равно 0, а другое нет
        if actual == 0 or forecast == 0:
            return 0  # Если есть расхождение (0 vs не 0) - точность 0%

        # Основная формула для случаев, когда оба значения не равны 0
        accuracy = (1 - abs(actual - forecast) / max(actual, forecast)) * 100
        return max(0, min(100, accuracy))

    # Расчет точности для всех товаров
    df['accuracy_percent'] = df.apply(calculate_accuracy, axis=1)

    # Первая метрика: средняя точность для всех товаров (исключая случаи 0-0)
    valid_accuracy_all = df['accuracy_percent'].dropna()
    avg_accuracy_all = valid_accuracy_all.mean() if len(valid_accuracy_all) > 0 else 0

    # Вторая метрика: средняя точность только для товаров с forecast_123_days_mean != 0
    non_zero_forecast = df[df['forecast_123_days_mean'] != 0]
    avg_accuracy_non_zero = non_zero_forecast['accuracy_percent'].mean()

    # Вывод результатов
    print("=" * 60)
    print("РЕЗУЛЬТАТЫ АНАЛИЗА ТОЧНОСТИ ПРОГНОЗА (ИСПРАВЛЕННЫЕ)")
    print("=" * 60)

    total_items = len(df)
    zero_zero_cases = len(df[(df['last_123_days_sum'] == 0) & (df['forecast_123_days_mean'] == 0)])
    valid_cases = len(valid_accuracy_all)

    print(f"\nВсего товаров: {total_items}")
    print(f"Товаров с ситуацией '0-0': {zero_zero_cases}")
    print(f"Товаров для расчета точности: {valid_cases}")
    print(f"Товаров с ненулевым прогнозом: {len(non_zero_forecast)}")

    print(f"\nСредняя точность прогноза (исключая 0-0): {avg_accuracy_all:.2f}%")
    print(f"Средняя точность прогноза для товаров с ненулевым прогнозом: {avg_accuracy_non_zero:.2f}%")

    # Детальная информация по каждому товару
    print(f"\nДетальная информация по товарам:")
    print("-" * 90)
    print(f"{'Артикул':<15} {'Факт':<10} {'Прогноз':<12} {'Точность':<10} {'Примечание':<15}")
    print("-" * 90)

    for _, row in df.iterrows():
        actual = row['last_123_days_sum']
        forecast = row['forecast_123_days_mean']
        accuracy = row['accuracy_percent']

        if pd.isna(accuracy):
            note = "0-0 (исключен)"
        elif accuracy == 0 and (actual == 0 or forecast == 0):
            note = "0 vs не-0"
        else:
            note = "расчет"

        accuracy_display = "исключен" if pd.isna(accuracy) else f"{accuracy:.1f}%"
        print(f"{row['Артикул']:<15} {actual:<10} {forecast:<12.1f} {accuracy_display:<10} {note:<15}")

    return avg_accuracy_all, avg_accuracy_non_zero, df


# Альтернативный вариант: считать 0-0 как 100%, но отдельно выводить статистику
def calculate_accuracy_metrics_v2(csv_file):
    df = pd.read_csv(csv_file)

    def calculate_accuracy(row):
        actual = row['last_123_days_sum']
        forecast = row['forecast_123_days_mean']

        # Вариант 2: считать 0-0 как 100%, но разделять статистику
        if actual == 0 and forecast == 0:
            return 100
        elif actual == 0 or forecast == 0:
            return 0
        else:
            accuracy = (1 - abs(actual - forecast) / max(actual, forecast)) * 100
            return max(0, min(100, accuracy))

    df['accuracy_percent'] = df.apply(calculate_accuracy, axis=1)

    # Раздельная статистика
    zero_zero_cases = df[(df['last_123_days_sum'] == 0) & (df['forecast_123_days_mean'] == 0)]
    non_zero_cases = df[(df['last_123_days_sum'] != 0) | (df['forecast_123_days_mean'] != 0)]
    non_zero_forecast = df[df['forecast_123_days_mean'] != 0]

    avg_all = df['accuracy_percent'].mean()
    avg_non_zero_only = non_zero_forecast['accuracy_percent'].mean()
    avg_without_zero_zero = non_zero_cases['accuracy_percent'].mean()

    print("=" * 60)
    print("АНАЛИЗ ТОЧНОСТИ (ВАРИАНТ 2 - С УЧЕТОМ 0-0)")
    print("=" * 60)
    print(f"Всего товаров: {len(df)}")
    print(f"Ситуаций '0-0': {len(zero_zero_cases)}")
    print(f"Остальных случаев: {len(non_zero_cases)}")
    print(f"Товаров с ненулевым прогнозом: {len(non_zero_forecast)}")
    print(f"\nСредняя точность (все товары, 0-0=100%): {avg_all:.2f}%")
    print(f"Средняя точность (без учета 0-0): {avg_without_zero_zero:.2f}%")
    print(f"Средняя точность (только ненулевой прогноз): {avg_non_zero_only:.2f}%")

    return avg_all, avg_without_zero_zero, avg_non_zero_only, df


# Использование скрипта
if __name__ == "__main__":
    csv_file_path = "combined_sales_forecast_report.csv"  # Замените на путь к вашему файлу

    try:
        print("ВЕРСИЯ 1: Исключаем случаи 0-0 из расчета")
        print("-" * 50)
        avg_all, avg_non_zero, result_df = calculate_accuracy_metrics(csv_file_path)

        print("\n" + "=" * 80)
        print("ВЕРСИЯ 2: Считаем случаи 0-0 как 100%")
        print("-" * 50)
        avg_all_v2, avg_no_zero_zero, avg_non_zero_v2, result_df_v2 = calculate_accuracy_metrics_v2(csv_file_path)

    except FileNotFoundError:
        print(f"Ошибка: Файл '{csv_file_path}' не найден")
    except Exception as e:
        print(f"Произошла ошибка: {e}")