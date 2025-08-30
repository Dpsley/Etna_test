import pandas as pd

# Загружаем CSV (пробуем ; как разделитель)
df = pd.read_csv("sales_full.csv", sep=",", dtype=str)

# Посмотрим первые строки и названия колонок
print("Колонки:", df.columns.tolist())
print(df.head(3))

# Если колонок без заголовков — pandas их назовёт 0,1,2,...
# тогда E это индекс 4
if len(df.columns) > 4:
    df.iloc[:, 4] = df.iloc[:, 4].fillna("Набор")
    df.iloc[:, 4] = df.iloc[:, 4].replace("", "Набор")
else:
    raise ValueError("В файле меньше 5 колонок, проверь разделитель!")

# Сохраняем
df.to_csv("sales_full_fixed.csv", sep=",", index=False)
print("Готово: sales_full_fixed.csv")
