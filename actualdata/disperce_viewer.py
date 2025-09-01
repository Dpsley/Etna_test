import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

# === Загрузка данных ===
df = pd.read_csv("expanded.csv", sep=";", parse_dates=["Date"])

# === Разброс продаж по артикулам ===
article_stats = (
    df.groupby("Article")["Sold"]
    .agg(["mean", "std", "sum"])
    .reset_index()
    .sort_values("std", ascending=False)
)

# Топ-5 товаров с самым большим разбросом
top_articles = article_stats.head(5)["Article"].tolist()
print("Топ-5 по разбросу:", top_articles)

# === Папка для графиков ===
output_dir = "graphs"
os.makedirs(output_dir, exist_ok=True)

# === Построение графиков ===
for art in top_articles:
    sub = df[df["Article"] == art].sort_values("Date")

    # --- Агрегируем по месяцам ---
    sub_month = sub.resample('MS', on='Date').sum().reset_index()  # MS = Month Start (первый день месяца)

    plt.figure(figsize=(12, 6))
    plt.plot(sub_month["Date"], sub_month["Sold"], linestyle='-', color='blue')

    plt.title(f"Продажи артикула {art} по месяцам")
    plt.xlabel("Дата (начало месяца)")
    plt.ylabel("Продажи за месяц")

    plt.grid(True)

    # Форматируем ось X по месяцам
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.MonthLocator())  # метка на каждый месяц
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))  # формат: Янв-2025

    plt.xticks(rotation=45)
    plt.tight_layout()

    filepath = os.path.join(output_dir, f"{art}_monthly.png")
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()

print(f"Графики месячных продаж сохранены в папке: {output_dir}")
