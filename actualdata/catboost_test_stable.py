import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import optuna
import os


# ----------------------
# БЕЗОПАСНЫЙ MAPE (объявляем ОДИН РАЗ!)
# ----------------------
def safe_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero = y_true != 0
    if not non_zero.any():
        return float('inf')  # Защита от всех нулевых значений
    return np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100


# ----------------------
# Загружаем данные
# ----------------------
df = pd.read_csv('expanded.csv', sep=';', parse_dates=['Date'], dayfirst=False)
df = df.sort_values('Date').reset_index(drop=True)  # КРИТИЧЕСКИ ВАЖНО

# ----------------------
# Фичи и таргет
# ----------------------
features = [
    "Department",
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
test = df[df["Date"] > "2025-07-20"]

X_train, y_train = train[features], train[target]
X_test, y_test = test[features], test[target]

# ----------------------
# Optuna: оптимизация ТОЛЬКО по RMSE
# ----------------------
tscv = TimeSeriesSplit(n_splits=10)


def objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 500, 4000),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
        'depth': trial.suggest_int('depth', 2, 16),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.01, 10.0, log=True),
        'random_strength': trial.suggest_float('random_strength', 0.1, 50.0),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 10.0),
        'border_count': trial.suggest_int('border_count', 32, 512),
        'one_hot_max_size': trial.suggest_int('one_hot_max_size', 1, 25),
        'loss_function': 'RMSE',
        'verbose': 0,
        'random_seed': 42,
        'task_type': 'GPU',
        'devices': '0'
    }

    rmse_scores = []
    for train_idx, val_idx in tscv.split(X_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        train_pool = Pool(X_tr, y_tr, cat_features=cat_features)
        val_pool = Pool(X_val, y_val, cat_features=cat_features)

        model = CatBoostRegressor(**params)
        model.fit(train_pool, eval_set=val_pool, verbose=False)
        y_pred = model.predict(X_val)

        # Считаем RMSE для текущего фолда
        rmse = mean_squared_error(y_val, y_pred) ** 0.5
        rmse_scores.append(rmse)

    # Возвращаем СРЕДНИЙ RMSE по всем фолдам
    return np.mean(rmse_scores)


# ----------------------
# Запускаем Optuna (ОДНА ЦЕЛЬ!)
# ----------------------
study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42), pruner=optuna.pruners.HyperbandPruner())
study.optimize(objective, n_trials=100)

print("✅ Лучшие параметры (оптимизированы по RMSE):")
print(study.best_params)

# ----------------------
# Финальная модель
# ----------------------
best_params = study.best_params
model = CatBoostRegressor(
    **best_params,
    task_type="GPU",
    devices='0',
    verbose=100
)
model.fit(
    Pool(X_train, y_train, cat_features=cat_features),
    eval_set=Pool(X_test, y_test, cat_features=cat_features),
    use_best_model=True
)

# ----------------------
# ПРОГНОЗ И МЕТРИКИ (все считаем НА ТЕСТЕ)
# ----------------------
y_pred = model.predict(X_test)

# Расчёт метрик с защитой от крайних случаев
rmse = mean_squared_error(y_test, y_pred) ** 0.5
mae = mean_absolute_error(y_test, y_pred)
mape = safe_mape(y_test, y_pred)
accuracy = max(0, 1 - mape / 100) if mape != float('inf') else 0
r2 = 1 - (np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))

# Выводим результаты в формате, понятном бизнесу
print("\n" + "=" * 50)
print(f"РЕЗУЛЬТАТЫ НА ТЕСТОВОМ НАБОРЕ (2025-07-21 и позже)")
print("=" * 50)
print(f"RMSE:          {rmse:.2f} шт.  → Средняя ошибка прогноза")
print(f"MAE:           {mae:.2f} шт.  → Среднее отклонение")
print(f"MAPE:          {mape:.2f}%    → Процентная ошибка")
print(f"Точность:      {accuracy:.2%}  → (1 - MAPE)")
print(f"Коэф. детерм.:  {r2:.4f}     → Объяснённая дисперсия")
print("=" * 50)

# ----------------------
# Сохраняем predictions
# ----------------------
pred_df = test[["Department", "Article", "ProductName", "Date"]].copy()
pred_df["Actual"] = y_test.values
pred_df["Predicted"] = y_pred.round().astype(int)
pred_df.to_csv("predictions.csv", sep=";", index=False)
print("\n✅ predictions.csv сохранён")

# ----------------------
# Сохраняем модель
# ----------------------
model.save_model("catboost_model.cbm")
print("✅ модель сохранена в catboost_model.cbm")

# ----------------------
# Анализ ошибок (ключевое для улучшения!)
# ----------------------
pred_df['error'] = (pred_df['Actual'] - pred_df['Predicted']).abs()
worst_products = pred_df.groupby(['Article', 'ProductName']).agg(
    avg_error=('error', 'mean'),
    max_error=('error', 'max'),
    sales_count=('Actual', 'count')
).sort_values('avg_error', ascending=False).head(10)

print("\n" + "=" * 50)
print("ТОП-10 ТОВАРОВ С НАИБОЛЬШИМИ ОШИБКАМИ")
print("=" * 50)
print(worst_products)
print("=" * 50)

# ----------------------
# Графики
# ----------------------
os.makedirs("plots", exist_ok=True)
for (dept, article, name), grp in pred_df.groupby(["Department", "Article", "ProductName"]):
    plt.figure(figsize=(12, 6))
    plt.plot(grp["Date"], grp["Actual"], 'o-', linewidth=2, markersize=6, label="Факт")
    plt.plot(grp["Date"], grp["Predicted"], 'x--', linewidth=1.5, markersize=5, label="Прогноз")

    # Добавляем зону ошибки ±10%
    plt.fill_between(
        grp["Date"],
        grp["Predicted"] * 0.9,
        grp["Predicted"] * 1.1,
        color='gray', alpha=0.2, label='Допустимая погрешность ±10%'
    )

    plt.title(f"{name} ({article}) [{dept}]\nRMSE: {mean_squared_error(grp['Actual'], grp['Predicted']) ** 0.5:.2f}",
              fontsize=14)
    plt.xlabel("Дата", fontsize=12)
    plt.ylabel("Продажи", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()

    safe_dept = "".join(c if c.isalnum() else "_" for c in dept)
    safe_article = "".join(c if c.isalnum() else "_" for c in article)
    plt.savefig(f"plots/{safe_dept}_{safe_article}.png", dpi=150)
    plt.close()

print("\n✅ PNG-графики сохранены в папку plots/")
print("   → Каждый график содержит зону допустимой погрешности ±10%")