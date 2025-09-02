import pandas as pd

# Загружаем CSV
df = pd.read_csv("predictions.csv", sep=";", parse_dates=["Date"])

# Сумма по каждому департаменту
grouped = df.groupby("Department").agg(
    fact_sum=("Actual", "sum"),
    forecast_sum=("Predicted", "sum")
).reset_index()

# Вывод
for _, row in grouped.iterrows():
    print(f"Департамент: {row['Department']}")
    print(f"  Сумма факт за месяц: {row['fact_sum']}")
    print(f"  Сумма прогноз за месяц: {row['forecast_sum']}")
    print("-" * 30)
