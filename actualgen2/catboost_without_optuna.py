#catboost_without_optuna.py

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Must be set before importing pyplot
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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
df = pd.read_csv('expanded.csv', sep=';', parse_dates=['Date'], dayfirst=False)

# ----------------------
# Фичи и таргет
# ----------------------
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

y_train = np.log1p(train[target].clip(lower=0))
X_train = train[features].copy()

finite_mask = np.isfinite(y_train)
if not finite_mask.all():
    removed = (~finite_mask).sum()
    print(f"⚠️ Удаляю {removed} строк(у) из train из-за NaN/inf.")
    X_train = X_train.loc[finite_mask].reset_index(drop=True)
    y_train = y_train[finite_mask].reset_index(drop=True)

X_test = test[features].copy()
y_test = test[target].copy()

# ----------------------
# Создаем Pool для CatBoost (безопаснее для больших данных)
# ----------------------
train_pool = Pool(data=X_train, label=y_train, cat_features=cat_features)
eval_pool  = Pool(data=X_test,  label=np.log1p(y_test.clip(lower=0)), cat_features=cat_features)

# ----------------------
# Финальная модель с фикс параметрами
# ----------------------
safe_params  = {
    'iterations': 1763,
    'learning_rate': 0.0364637585556757,
    'depth': 6,
    'l2_leaf_reg': 0.02377294218983323,
    'random_strength': 1.161435457162796,
    'bagging_temperature': 3.9589913674642148,
    'border_count': 450,
    'one_hot_max_size': 4,
    'loss_function': 'RMSE',
    'random_seed': 42
}

model = CatBoostRegressor(**safe_params, verbose=100)
model.fit(
    train_pool,
    eval_set=eval_pool,
    early_stopping_rounds=100,
    use_best_model=True
)

# ----------------------
# Прогноз и метрики
# ----------------------
y_pred_log = model.predict(eval_pool)
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
for i, ((dept, article, name), grp) in enumerate(pred_df.groupby(["Department","Article","ProductName"])):
    plt.figure(figsize=(10, 5))
    plt.plot(grp["Date"], grp["Actual"], marker="o", label="Факт")
    plt.plot(grp["Date"], grp["Predicted"], marker="x", label="Прогноз")
    plt.fill_between(grp["Date"], grp["Predicted"]*0.9, grp["Predicted"]*1.1, color="gray", alpha=0.2, label="±10%")
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