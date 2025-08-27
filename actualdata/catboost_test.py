import pandas as pd
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import os

# Загружаем
df = pd.read_csv("expanded.csv", sep=";")

# Фичи и таргет
features = [
    "sold_lag1", "sold_lag7", "sold_ma7", "sold_ma14",
    "stock_diff", "restock_flag", "days_since_last_restock",
    "day_of_week", "is_weekend", "month"
]
target = "Sold"

# Train/Test (по дате!)
train = df[df["Date"] <= "2025-06-30"]
test  = df[df["Date"] >  "2025-06-30"]

X_train, y_train = train[features], train[target]
X_test, y_test   = test[features],  test[target]

# Модель
model = CatBoostRegressor(
    verbose=1000,
    iterations=2000,
    depth=16,
    learning_rate=0.03,
    random_seed=42,
    task_type="GPU",
    loss_function="MAE",
    devices="0"
)
model.fit(X_train, y_train)

# Прогноз
y_pred = model.predict(X_test)

# MAE
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MAPE:", mean_absolute_percentage_error(y_test, y_pred))

# Сохраняем в predictions.csv
pred_df = test[["Article", "ProductName", "Date"]].copy()
pred_df["Actual"] = y_test.values
pred_df["Predicted"] = y_pred.round().astype(int)
pred_df.to_csv("predictions.csv", sep=";", index=False)
print("✅ predictions.csv сохранён")

model.save_model("catboost_model.cbm")
print("✅ модель сохранена в catboost_model.cbm")
# Директория для графиков
os.makedirs("plots", exist_ok=True)

# Графики по каждому товару
for (article, name), grp in pred_df.groupby(["Article", "ProductName"]):
    plt.figure(figsize=(10, 5))
    plt.plot(grp["Date"], grp["Actual"], marker="o", label="Факт")
    plt.plot(grp["Date"], grp["Predicted"], marker="x", label="Прогноз")
    plt.title(f"{name} ({article})")
    plt.xlabel("Дата")
    plt.ylabel("Продажи")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"plots/{article}.png")
    plt.close()

print("✅ PNG-графики сохранены в папку plots/")
