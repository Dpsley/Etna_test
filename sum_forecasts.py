import json
from datetime import datetime

# допустим, твой JSON лежит в файле data.json
with open("forecast.json", "r", encoding="utf-8") as f:
    data = json.load(f)

results = {}

for article, records in data.items():
    july_sum = sum(
        rec["target"]
        for rec in records
        if datetime.fromisoformat(rec["timestamp"]).month == 7
    )
    results[article] = july_sum

print("Суммы за июль по артикулам:")
for art, total in results.items():
    print(f"{art}: {total}")
