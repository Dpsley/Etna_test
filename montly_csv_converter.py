import pandas as pd
import re

df = pd.read_csv("monthly.csv", sep=";")

fact_cols = [c for c in df.columns if c.startswith("Факт")]

result = []

for _, row in df.iterrows():
    vid = row["Вид"]
    name = row["Номенклатура"]
    article = row["Артикул"]

    for col in fact_cols:
        val = row[col]
        if pd.isna(val):
            continue

        target = int(str(val).replace(" ", ""))

        # Факт2_11 → year=2023 + (2-1) = 2024, month=11
        m = re.search(r"Факт(\d+)_(\d+)", col)
        year_offset = int(m.group(1)) - 1
        month = int(m.group(2))
        year = 2023 + year_offset

        timestamp = f"{year}-{month:02d}"

        result.append({
            "ProductType": vid,
            "ProductName": name,
            #"артикул": article,
            "target": target,
            "timestamp": timestamp,
            "segment": article,
        })

out = pd.DataFrame(result)
out.to_csv("output_monthly.csv", index=False)
print("Готово. output.csv сформирован.")
