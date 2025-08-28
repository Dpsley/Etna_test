import pandas as pd
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor, Pool, cv
from sklearn.metrics import mean_absolute_error, mean_squared_error
import optuna
import os

# ----------------------
# Загружаем данные
# ----------------------
df = pd.read_csv('expanded.csv', sep=';', parse_dates=['Date'], dayfirst=False)

# ----------------------
# Фичи и таргет
# ----------------------
features = [
    "Department",   # категориальная
    "sold_lag1", "sold_lag7", "sold_ma7", "sold_ma14",
    "stock_diff", "restock_flag", "days_since_last_restock",
    "Reserve", "Available",
    "day_of_week", "is_weekend", "month"
]
cat_features = ["Department"]
target = "Sold"

# ----------------------
# Train/Test по дате
# ----------------------
train = df[df["Date"] <= "2025-07-20"]
test  = df[df["Date"] >  "2025-07-20"]

X_train, y_train = train[features], train[target]
X_test, y_test   = test[features],  test[target]

# ----------------------
# Optuna: оптимизация параметров
# ----------------------
def objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 1000, 3000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'depth': trial.suggest_int('depth', 4, 16),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.1, 10.0, log=True),
        'random_strength': trial.suggest_float('random_strength', 0.1, 10.0),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 10.0),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'one_hot_max_size': trial.suggest_int('one_hot_max_size', 2, 25),
        'loss_function': 'RMSE',
        'verbose': 0,
        'random_seed': 42
    }

    pool = Pool(X_train, y_train, cat_features=cat_features)

    cv_data = cv(
        params=params,
        pool=pool,
        fold_count=5,
        partition_random_seed=42,
        verbose=False
    )
    return cv_data['test-RMSE-mean'].min()

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=30)

print("✅ Оптимальные параметры:")
print(study.best_params)

# ----------------------
# Финальная модель с лучшими параметрами
# ----------------------
best_params = study.best_params
model = CatBoostRegressor(**best_params, verbose=100)
model.fit(X_train, y_train, cat_features=cat_features)

# ----------------------
# Прогноз и метрики
# ----------------------
y_pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred)**0.5)
print("R2:", model.score(X_test, y_test))

# ----------------------
# Сохраняем predictions
# ----------------------
pred_df = test[["Department","Article","ProductName","Date"]].copy()
pred_df["Actual"] = y_test.values
pred_df["Predicted"] = y_pred.round().astype(int)
pred_df.to_csv("predictions.csv", sep=";", index=False)
print("✅ predictions.csv сохранён")

# ----------------------
# Сохраняем модель
# ----------------------
model.save_model("catboost_model.cbm")
print("✅ модель сохранена в catboost_model.cbm")

# ----------------------
# Директория для графиков
# ----------------------
os.makedirs("plots", exist_ok=True)

# Графики по Department + Article + ProductName
for (dept, article, name), grp in pred_df.groupby(["Department","Article","ProductName"]):
    plt.figure(figsize=(10, 5))
    plt.plot(grp["Date"], grp["Actual"], marker="o", label="Факт")
    plt.plot(grp["Date"], grp["Predicted"], marker="x", label="Прогноз")
    plt.title(f"{name} ({article}) [{dept}]")
    plt.xlabel("Дата")
    plt.ylabel("Продажи")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"plots/{dept}_{article}.png")
    plt.close()

print("✅ PNG-графики сохранены в папку plots/")
