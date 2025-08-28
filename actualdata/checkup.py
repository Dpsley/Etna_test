import pandas as pd

# читаем с правильным разделителем
preds = pd.read_csv("predictions.csv", sep=";")

# считаем ошибки
preds["abs_error"] = (preds["Actual"] - preds["Predicted"]).abs()
preds["sq_error"] = (preds["Actual"] - preds["Predicted"])**2

# топ-10 по MAE
top_abs = preds.sort_values("abs_error", ascending=False).head(10)

# топ-10 по RMSE (квадратичная ошибка)
top_sq = preds.sort_values("sq_error", ascending=False).head(10)

print("🔴 Топ-10 самых больших ошибок по MAE:")
print(top_abs[["Article", "ProductName", "Date", "Actual", "Predicted", "abs_error"]])

print("\n🔴 Топ-10 выбросов по RMSE:")
print(top_sq[["Article", "ProductName", "Date", "Actual", "Predicted", "sq_error"]])
