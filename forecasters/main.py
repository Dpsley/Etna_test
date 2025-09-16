import json
from pathlib import Path

import pandas as pd
from etna.pipeline import Pipeline
from datasets.main import load_actual_dataset
from graphs.main import new_forecast_plot
from pipelines.main import load_pipline_from_dump


def new_forecast(pipeline: Pipeline):
    forecast = pipeline.forecast(prediction_interval=True, ts=load_actual_dataset())
    print("forecast", forecast)

    forecast_df = forecast.to_pandas(flatten=True).reset_index()
    forecast_df["timestamp"] = pd.to_datetime(forecast_df["timestamp"])
    forecast_df["department"] = forecast_df["segment"].str.split("|").str[0]
    forecast_df["article"] = forecast_df["segment"].str.split("|").str[1]
    forecast_agg = forecast_df.groupby(["timestamp", "department", "article"], as_index=False)["target"].sum()
    print("=== Прогноз по департаментам и артикулам ===")
    forecast_summary = forecast_agg.groupby(["department", "article"], as_index=False)["target"].sum()
    print(forecast_summary.sort_values(["department", "article"]).to_string(index=False))
    # проверим какие колонки реально есть
    has_lower = "target_0.025" in forecast_df.columns
    has_upper = "target_0.975" in forecast_df.columns

    result = {}
    for seg in forecast.segments:
        cols = ["timestamp", "target"]
        if has_lower:
            cols.append("target_0.025")
        if has_upper:
            cols.append("target_0.975")

        seg_df = forecast_df[forecast_df["segment"] == seg][cols]

        result[seg] = [
            {
                "timestamp": row["timestamp"].strftime("%Y-%m-%d"),
                "target": float(row["target"]),
                "target_lower": float(row["target_0.025"]) if has_lower else None,
                "target_upper": float(row["target_0.975"]) if has_upper else None,
            }
            for _, row in seg_df.iterrows()
        ]

    json_str = json.dumps(result, ensure_ascii=False, indent=2)
    # сохранить в файл
    output_file = Path("forecast.json")
    output_file.write_text(json_str, encoding="utf-8")

    print(f"✅ JSON сохранён в {output_file.resolve()}")
    #print(json_str)

    # график на основе forecast_df
    new_forecast_plot(forecast_df)

new_forecast(load_pipline_from_dump())