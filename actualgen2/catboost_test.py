import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import optuna
import os

# ----------------------
# safe MAPE
# ----------------------
def safe_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero = y_true != 0
    if not non_zero.any():
        return float("inf")
    return np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100

# ----------------------
# Загружаем данные
# ----------------------
df = pd.read_csv(
    'expanded.csv',
    sep=';',
    parse_dates=['Date'],
    dayfirst=False
)

product_features = [
    "ProductGroup","ProductType","Вид","Регион","Тип",
    "Марка","Код производителя","Вкус","Ароматизированный",
    "brand_region","type_flavor"
]
lag_days = [1,2,3,7,14,21,28,60,90]
sold_std = [7, 30]
ma_windows = [3,7,14,21,28,60,90]

features = product_features + [
    "Department",
    *[f'sold_lag{l}' for l in lag_days],
    *[f'sold_ma{w}' for w in ma_windows],
    *[f'sold_median{w}' for w in ma_windows],
    *[f'sold_std{d}' for d in sold_std],
    "sold_last_day_binary","sold_last_week_binary","sold_last_month_binary",
    "day_of_week","is_weekend","month","day",
    "quarter","is_start_of_month","is_end_of_month"
]
cat_features = ["Department"] + product_features
for col in cat_features:
    df[col] = df[col].fillna("NA").astype(str)
target = "Sold"

df['Sold'] = pd.to_numeric(df['Sold'], errors='coerce').fillna(0).astype(int)

df['Sold'] = df['Sold'].clip(lower=0)

# ----------------------
# Train/Test
# ----------------------
train = df[df["Date"] <= "2025-06-30"].copy()
test  = df[df["Date"] >  "2025-06-30"].copy()

# Безопасный таргет: лог от (>=0)
y_train = np.log1p(train[target].clip(lower=0))
X_train = train[features].copy()

# Удаляем строки с невалидным таргетом (NaN / inf), чтобы CatBoost не ругался
finite_mask = np.isfinite(y_train)
if not finite_mask.all():
    removed = (~finite_mask).sum()
    print(f"⚠️ Удаляю {removed} строк(у) из train из-за невалидного таргета (NaN/inf).")
    X_train = X_train.loc[finite_mask].reset_index(drop=True)
    y_train = y_train[finite_mask].reset_index(drop=True)

# Аналогично для теста (если планируешь лог-метрику/сравнение)
# На тесте ты у себя не логируешь y_test в предсказаниях — оставлю как у тебя:
X_test = test[features].copy()
y_test = test[target].copy()

# ----------------------
# Optuna оптимизация
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
        'random_seed': 42
    }

    tscv = TimeSeriesSplit(n_splits=10)
    rmse_scores = []

    for train_idx, val_idx in tscv.split(X_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model = CatBoostRegressor(**params)
        model.fit(
            X_tr, y_tr,
            cat_features=cat_features,
            eval_set=(X_val, y_val),
            early_stopping_rounds=200,
            verbose=100
        )
        preds = model.predict(X_val)
        rmse_scores.append(mean_squared_error(y_val, preds))

    return np.mean(rmse_scores)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50, show_progress_bar=True)
best_params = study.best_params
print("✅ Оптимальные параметры:", best_params)

# ----------------------
# Финальная модель
# ----------------------
model = CatBoostRegressor(**best_params, verbose=100)
model.fit(
    X_train, y_train,
    cat_features=cat_features,
    eval_set=(X_test, np.log1p(y_test.clip(lower=0))),
    early_stopping_rounds=200,
    use_best_model=True
)

# ----------------------
# Прогноз и метрики
# ----------------------
y_pred_log = model.predict(X_test)
y_pred = np.expm1(y_pred_log)

mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred)
mape = safe_mape(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("="*40)
print(f"MAE:   {mae:.2f}")
print(f"RMSE:  {rmse:.2f}")
print(f"MAPE:  {mape:.2f}%")
print(f"R2:    {r2:.4f}")
print("="*40)

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
# Графики
# ----------------------
os.makedirs("plots", exist_ok=True)
for (dept, article, name), grp in pred_df.groupby(["Department","Article","ProductName"]):
    plt.figure(figsize=(10, 5))
    plt.plot(grp["Date"], grp["Actual"], marker="o", label="Факт")
    plt.plot(grp["Date"], grp["Predicted"], marker="x", label="Прогноз")
    plt.fill_between(
        grp["Date"],
        grp["Predicted"]*0.9,
        grp["Predicted"]*1.1,
        color="gray", alpha=0.2, label="±10%"
    )
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
