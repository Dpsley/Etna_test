import pandas as pd

# список файлов
files = [
    "qu_CIMPCH-000062.csv",
    "qu_TALTHA-DP0061.csv",
    "qu_TALTHA-DP0082.csv"
]

# читаем и объединяем
dfs = [pd.read_csv(f, sep=",", quotechar='"', parse_dates=["Date"], dayfirst=True) for f in files]
combined_df = pd.concat(dfs, ignore_index=True)

# сохраняем в один CSV
combined_df.to_csv("combined.csv", sep=",", index=False, quotechar='"')
print("Saved combined CSV:", combined_df.shape)