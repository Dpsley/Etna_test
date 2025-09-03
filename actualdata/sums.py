import pandas as pd

# Загружаем CSV
df = pd.read_csv("predictions.csv", sep=";", parse_dates=["Date"])

# Группировка: подразделение + артикул + наименование товара
grouped = df.groupby(
    ["Department", "Article", "ProductName"], as_index=False
).agg(
    план=("Predicted", "sum"),
    факт=("Actual", "sum")
)

# Переименовываем колонки как надо
grouped = grouped.rename(columns={
    "Department": "подразделение",
    "Article": "наименование товара (код)",
    "ProductName": "наименование товара"
})

# Сохраняем в CSV
grouped.to_csv("summary.csv", sep=";", index=False)

print("✅ summary.csv готов")
