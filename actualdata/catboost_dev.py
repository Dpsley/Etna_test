import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import os

# ----------------------
# Load
# ----------------------
df = pd.read_csv('expanded.csv', sep=';', parse_dates=['Date'], dayfirst=False)

# ----------------------
# Features / target
# ----------------------
features = [
    "Department",   # categorical
    "sold_lag1", "sold_lag7", "sold_ma7", "sold_ma14",
    "stock_diff", "restock_flag", "days_since_last_restock",
    "Reserve", "Available",
    "day_of_week", "is_weekend", "month"
]
cat_features = ["Department"]
target = "Sold"

# ----------------------
# Train / test by date
# ----------------------
split_date = "2025-07-20"
train = df[df["Date"] <= split_date].copy()
test  = df[df["Date"] >  split_date].copy()

if train.empty:
    raise SystemExit(f"Train is empty, check split date {split_date}")
if test.empty:
    print("WARN: test is empty — ничего прогнозить не на что")

X_train_all, y_train_all = train[features], train[target]
X_test, y_test = test[features], test[target]

# ----------------------
# Split train_inner / val_inner (для Optuna)
# ----------------------
train_sorted = train.sort_values("Date")
val_cut = int(len(train_sorted) * 0.8)
train_inner = train_sorted.iloc[:val_cut]
val_inner   = train_sorted.iloc[val_cut:]

X_tr, y_tr = train_inner[features], train_inner[target]
X_val, y_val = val_inner[features], val_inner[target]

# ----------------------
# Оптимизация (Optuna)
# ----------------------
def objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 500, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'depth': trial.suggest_int('depth', 4, 12),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.1, 10.0, log=True),
        'random_strength': trial.suggest_float('random_strength', 0.1, 10.0),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 3.0),
        'border_count': trial.suggest_int('border_count', 32, 128),
        'one_hot_max_size': trial.suggest_int('one_hot_max_size', 2, 10),
        'loss_function': 'RMSE',
        'task_type': 'GPU',
        'devices': '0',
        'verbose': 0,
        'random_seed': 42
    }

    model = CatBoostRegressor(**params)
    model.fit(
        X_tr, y_tr,
        eval_set=(X_val, y_val),
        cat_features=cat_features,
        use_best_model=True,
        early_stopping_rounds=50,
        verbose=False
    )
    y_val_pred = model.predict(X_val)
    rmse = mean_squared_error(y_val, y_val_pred)
    return rmse

study = optuna.create_study(
    direction='minimize',
    sampler=TPESampler(seed=42),
    pruner=MedianPruner(n_startup_trials=5)
)
study.optimize(objective, n_trials=80)

print("✅ Best params:", study.best_params)

# ----------------------
# Финальное обучение на train_inner + val_inner
# ----------------------
best_params = study.best_params.copy()
best_params.setdefault('iterations', 1000)
best_params.setdefault('loss_function', 'RMSE')
best_params.update({
    'verbose': 100,
    'random_seed': 42,
    'od_type': 'Iter',
    'od_wait': 50,
    'task_type': 'GPU',
    'devices': '0'
})

final_model = CatBoostRegressor(**best_params)
final_model.fit(
    X_tr, y_tr,
    eval_set=(X_val, y_val),
    cat_features=cat_features,
    use_best_model=True,
    early_stopping_rounds=50
)

# ----------------------
# Predict & metrics
# ----------------------
y_pred = final_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred)
eps = 1e-8
mape = np.mean(np.abs((y_test.values - y_pred) / np.maximum(eps, y_test.values)))

print("MAE:", mae)
print("RMSE:", rmse)
print("R2:", final_model.score(X_test, y_test))
print("MAPE (safe):", mape)
print("Accuracy (1 - MAPE):", 1 - mape)

# ----------------------
# Save predictions
# ----------------------
pred_df = test[["Department","Article","ProductName","Date"]].copy()
pred_df["Actual"] = y_test.values
pred_df["Predicted"] = np.clip(np.round(y_pred), 0, None).astype(int)
pred_df.to_csv("predictions.csv", sep=";", index=False)
print("✅ predictions.csv saved")

# ----------------------
# Save model
# ----------------------
final_model.save_model("catboost_model.cbm")
print("✅ model saved")

# ----------------------
# Plots
# ----------------------
os.makedirs("plots", exist_ok=True)

def safe_fname(s):
    s = str(s)
    s = re.sub(r'[^0-9a-zA-Z\-_.]+', '_', s)
    return s[:120]

for (dept, article, name), grp in pred_df.groupby(["Department","Article","ProductName"]):
    if len(grp) < 2:
        continue
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
    fname = f"plots/{safe_fname(dept)}_{safe_fname(article)}.png"
    plt.savefig(fname, bbox_inches='tight')
    plt.close()

print("✅ PNGs saved to plots/")
