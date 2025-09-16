import os
import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv
from pandas import DataFrame

load_dotenv()
PLOT_DIR = os.getenv("FORECAST_PLOT_DIR")

def new_forecast_plot(forecast_df: DataFrame):
    """
    Создаёт графики прогноза для каждого сегмента (department+article).

    Parameters
    ----------
    forecast_df : pd.DataFrame
        Должен содержать колонки ['timestamp', 'department', 'article', 'target']
    """
    os.makedirs(PLOT_DIR, exist_ok=True)
    forecast_df["timestamp"] = pd.to_datetime(forecast_df["timestamp"])
    forecast_df["department"] = forecast_df["segment"].str.split("|").str[0]
    forecast_df["article"] = forecast_df["segment"].str.split("|").str[1]
    for (dept, art), forecast_grp in forecast_df.groupby(["department", "article"]):
        if forecast_grp.empty:
            continue

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(forecast_grp["timestamp"], forecast_grp["target"],
                label="Forecast (план)", color="red", linestyle="--", linewidth=2)

        ax.set_title(f"Forecast: {dept} | {art}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Target")
        ax.legend()
        plt.tight_layout()

        filename = f"{PLOT_DIR}/{dept}_{art}.png"
        plt.savefig(filename, dpi=150)
        plt.close()

        print(f"Saved plot: {os.path.abspath(filename)}")
