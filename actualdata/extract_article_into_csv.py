import pandas as pd
import os

ARTICLE = "TALTHB-DP0031"
OUTPUT_DIR = "by_department"

# Создаём папку для файлов
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Загружаем полный CSV
df = pd.read_csv("expanded.csv", sep=";", parse_dates=["Date"], dayfirst=False)

# Фильтруем по Article
df_filtered = df[df["Article"] == ARTICLE].copy()

if df_filtered.empty:
    print(f"⚠️ Нет данных для Article={ARTICLE}")
else:
    # Проходим по каждому Department
    for dept, grp in df_filtered.groupby("Department"):
        filename = f"{OUTPUT_DIR}/expanded_{ARTICLE}_{dept}.csv"
        grp.to_csv(filename, sep=";", index=False)
        print(f"✅ Сохранено {len(grp)} строк в {filename}")
