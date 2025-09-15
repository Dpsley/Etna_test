import pandas as pd

# путь к исходному файлу
DATA_CSV = "expanded_etna.csv"

# сегмент, который нужен
SEGMENT_FILTER = "АТ Москва|TALTHA-BP0026"  # или используй regex "АТ Москва|TALTHA-BP0026"

# путь к новому файлу
OUTPUT_CSV = "filtered_segment.csv"

# читаем CSV
df = pd.read_csv(DATA_CSV)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values(['segment', 'timestamp']).reset_index(drop=True)

# фильтруем только нужный сегмент
df_filtered = df[df['segment'] == SEGMENT_FILTER].copy()

# Сохраняем в CSV
df_filtered.to_csv(OUTPUT_CSV, index=False)

print(f"Сегмент {SEGMENT_FILTER} сохранён в {OUTPUT_CSV}, строк: {len(df_filtered)}")