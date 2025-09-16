import json
from pathlib import Path

# --- Настройка ---
INPUT_FILE = Path("forecast.json")  # Укажите путь к вашему файлу

# --- Загрузка данных ---
with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    data = json.load(f)

# --- Подсчет сумм ---
segment_sums = {}
for segment_name, records in data.items():
    total = sum(record["target"] for record in records)
    segment_sums[segment_name] = total

# --- Вывод результатов ---
print("=" * 60)
print("СУММА ПРОГНОЗОВ (target) ПО СЕГМЕНТАМ")
print("=" * 60)

for segment, total in segment_sums.items():
    print(f"{segment:<40} : {total:>15,.2f}")

print("-" * 60)
grand_total = sum(segment_sums.values())
print(f"{'ИТОГО':<40} : {grand_total:>15,.2f}")
