import pandas as pd

# Загружаем
df = pd.read_csv("data.csv", sep=",", dayfirst=True, parse_dates=["Date"])

# Чистим ненужное
for col in ["Stock", "Reserve", "Available"]:
    if col in df.columns:
        df.drop(columns=[col], inplace=True)

if len(df.columns) > 4:
    df.iloc[:, 4] = df.iloc[:, 4].fillna("Набор")
    df.iloc[:, 4] = df.iloc[:, 4].replace("", "Набор")
else:
    raise ValueError("В файле меньше 5 колонок, проверь разделитель!")

# Переименуем для ETNA
df = df.rename(columns={"Date": "timestamp", "Sold": "target"})

# Добавим segment = Department|Article
df["segment"] = df["Department"].astype(str) + "|" + df["Article"].astype(str)

# Сохраним в новый файл
df.to_csv("expanded_etna.csv", sep=",", index=False)
print("Готово: expanded_etna.csv")
