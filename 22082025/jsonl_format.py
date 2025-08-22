import pandas as pd
import json

# Читаем CSV (разделитель ; и даты как строки)
df = pd.read_csv("test_new.csv", sep=";", parse_dates=["Date"], dayfirst=True)

# Преобразуем каждую строку в JSON и пишем в .jsonl
with open("sales.jsonl", "w", encoding="utf-8") as f:
    for _, row in df.iterrows():
        # Если нужно, свойства товара превращаем из строки в dict
        try:
            row["ProductProperties"] = json.loads(row["ProductProperties"])
        except json.JSONDecodeError:
            pass
        row["Date"] = row["Date"].strftime("%Y-%m-%d")

        f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
