import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import optuna
import os
from collections import deque

# ----------------------
# Параметры
# ----------------------
ARTICLE = "TALTHB-DP0031"
SPLIT_DATE = "2025-07-20"
N_FUTURE_DAYS = 30
FEATURES = [
    "Department",
    "sold_lag1", "sold_lag7", "sold_ma7", "sold_ma14",
    "stock_diff", "restock_flag", "days_since_last_restock",
    "Reserve", "Available",
    "day_of_week", "is_weekend", "month"
]
CAT_FEATURES = ["Department"]
TARGET = "Sold"

# ----------------------
# Загружаем данные
# ----------------------
df = pd.read_csv('expanded.csv', sep=';', parse_dates=['Date'], dayfirst=False)

# ----------------------
# Фильтруем по артикулу
# ----------------------
df = df[df["Article"] == ARTICLE].copy()
if df.empty:
    raise SystemExit(f"Нет данных для Article={ARTICLE}")

# ----------------------
# Train/Test split
# ----------------------
train = df[df["Date"] <= SPLIT_DATE].sort_values("Date")
test  = df[df["Date"] >  SPLIT_DATE].sort_values("Date")

X_train, y_train = train[FEATURES], train[TARGET]
X_test, y_test   = test[FEATURES], test[TARGET]

# ----------------------
# Optuna: оптимизация параметров
# ----------------------
def objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 500, 3000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.6, log=True),
        'depth': trial.suggest_int('depth', 4, 16),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.01, 10.0, log=True),
        'random_strength': trial.suggest_float('random_strength', 0.1, 20.0),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 5.0),
        'border_count': trial.suggest_int('border_count', 32, 512),
        'one_hot_max_size': trial.suggest_int('one_hot_max_size', 2, 25),
        'loss_function': 'RMSE',
        'random_seed': 42,
        'verbose': 0
    }

    tscv = TimeSeriesSplit(n_splits=5)
    rmse_scores = []

    for train_idx, val_idx in tscv.split(X_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model = CatBoostRegressor(**params, task_type='GPU', devices='0')
        model.fit(
            X_tr, y_tr,
            cat_features=CAT_FEATURES,
            eval_set=(X_val, y_val),
            early_stopping_rounds=50,
            verbose=100
        )
        rmse_scores.append(mean_squared_error(y_val, model.predict(X_val))**0.5)

    return np.mean(rmse_scores)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50, show_progress_bar=True)

print("✅ Оптимальные параметры:")
print(study.best_params)

# ----------------------
# Финальная модель с лучшими параметрами
# ----------------------
best_params = study.best_params
model = CatBoostRegressor(**best_params, verbose=100)
model.fit(X_train, y_train, cat_features=CAT_FEATURES, eval_set=(X_test, y_test), early_stopping_rounds=100, use_best_model=True)

# ----------------------
# Прогноз на тест + N_FUTURE_DAYS
# ----------------------
all_dates = pd.concat([
    test["Date"],
    pd.Series(pd.date_range(start=test["Date"].max() + pd.Timedelta(days=1), periods=N_FUTURE_DAYS))
]).reset_index(drop=True)

last_row = train.iloc[-1].copy()
preds = []

lag1 = last_row["sold_lag1"]
lag7_window = deque([last_row["sold_lag7"]] + [lag1]*6, maxlen=7)
ma7_window = deque([last_row["sold_ma7"]]*7, maxlen=7)
ma14_window = deque([last_row["sold_ma14"]]*14, maxlen=14)
available_stock = last_row["Available"]

for date in all_dates:
    if last_row["restock_flag"] == 1:
        available_stock += last_row["stock_diff"]

    X_pred = last_row[FEATURES].to_frame().T
    X_pred["Available"] = available_stock

    y_hat = model.predict(X_pred)[0]
    y_hat = min(y_hat, available_stock)
    preds.append(y_hat)

    available_stock -= y_hat
    if available_stock < 0: available_stock = 0

    lag1 = y_hat
    lag7_window.append(y_hat)
    ma7_window.append(y_hat)
    ma14_window.append(y_hat)

    last_row["sold_lag1"] = lag1
    last_row["sold_lag7"] = lag7_window[0]
    last_row["sold_ma7"] = np.mean(ma7_window)
    last_row["sold_ma14"] = np.mean(ma14_window)
    last_row["Available"] = available_stock

    last_row["Date"] = date
    last_row["day_of_week"] = last_row["Date"].weekday()
    last_row["is_weekend"] = int(last_row["day_of_week"] >= 5)
    last_row["month"] = last_row["Date"].month

forecast_df = pd.DataFrame({
    "Date": all_dates,
    "Predicted": np.round(preds).astype(int)
})
forecast_df.to_csv(f"forecast_{ARTICLE}_stock.csv", sep=";", index=False)
print(f"✅ Forecast saved to forecast_{ARTICLE}_stock.csv")

# ----------------------
# График
# ----------------------
plt.figure(figsize=(10, 5))
plt.plot(test["Date"], test["Sold"], marker="o", label="Факт (тест)")
plt.plot(forecast_df["Date"], forecast_df["Predicted"], marker="x", label="Прогноз")
plt.title(f"Прогноз продаж {ARTICLE} с учетом остатков")
plt.xlabel("Дата")
plt.ylabel("Продажи")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
os.makedirs("plots", exist_ok=True)
plt.savefig(f"plots/forecast_{ARTICLE}_stock.png")
plt.close()
print("✅ PNG saved")
