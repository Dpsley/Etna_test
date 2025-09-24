import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

DATA_CSV = os.getenv("MAIN_CSV_SRC")


def find_invalid_targets():
    # Читаем CSV файл с явным указанием типов для проблемных колонок
    df = pd.read_csv(DATA_CSV, dtype={'target': str}, low_memory=False)

    print(f"Всего строк в файле: {len(df)}")
    print(f"Тип данных колонки target: {df['target'].dtype}")
    print(f"Уникальные значения в target: {df['target'].unique()[:20]}")  # Покажем первые 20 уникальных значений

    # Проверим наличие NaN/пустых значений
    nan_count = df['target'].isna().sum()
    empty_count = (df['target'] == '').sum()
    print(f"NaN значений в target: {nan_count}")
    print(f"Пустых строк в target: {empty_count}")

    # Функция для проверки валидности значения target
    def is_valid_target(x):
        if pd.isna(x) or x == '' or x is None:
            return False

        # Пробуем преобразовать в число
        try:
            num = float(str(x).strip())
            # Проверяем, что это целое число и не отрицательное
            return num >= 0 and num.is_integer() and not pd.isna(num)
        except (ValueError, TypeError):
            return False

    # Применяем функцию проверки
    valid_mask = df['target'].apply(is_valid_target)

    # Находим строки с невалидными значениями
    invalid_rows = df[~valid_mask]

    print(f"\nНайдено строк с невалидными значениями target: {len(invalid_rows)}")

    if len(invalid_rows) > 0:
        print("\nСтроки с невалидными значениями target:")
        print(invalid_rows[['department', 'productName', 'timestamp', 'target']].head(
            20))  # Покажем только первые 20 и ключевые колонки

        # Детальная информация о невалидных значениях
        print("\nДетальная информация о невалидных значениях:")
        invalid_values = invalid_rows['target'].value_counts()
        for value, count in invalid_values.items():
            print(f"Значение '{value}': {count} раз(а)")

        # Сохраняем результат в файл
        invalid_rows.to_csv('invalid_targets.csv', index=False)
        print(f"\nРезультат сохранен в файл: invalid_targets.csv")
    else:
        print("Все значения в колонке target являются целыми положительными числами!")


find_invalid_targets()