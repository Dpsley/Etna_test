import json
from pathlib import Path

import pandas as pd
from etna.pipeline import Pipeline
from datasets.main import load_actual_dataset
from graphs.main import new_forecast_plot
from pipelines.main import load_pipline_from_dump


def new_forecast(pipeline: Pipeline):
    forecast = pipeline.forecast(prediction_interval=True, quantiles=[0.1, 0.95])
    print("forecast", forecast)

    forecast_df = forecast.to_pandas(flatten=True).reset_index()
    forecast_df["timestamp"] = pd.to_datetime(forecast_df["timestamp"])
    forecast_df["department"] = forecast_df["segment"].str.split("|").str[0]
    forecast_df["article"] = forecast_df["segment"].str.split("|").str[1]

    # проверим, есть ли колонки квантилей
    has_lower = "target_0.1" in forecast_df.columns
    has_upper = "target_0.95" in forecast_df.columns

    # создаём колонку для квантилей, обрезаем отрицательные значения
    if has_lower:
        forecast_df["target_0.1"] = forecast_df["target_0.1"].apply(lambda x: max(float(x), 0))
    if has_upper:
        forecast_df["target_0.95"] = forecast_df["target_0.95"].apply(lambda x: max(float(x), 0))

    # агрегация по департаментам и артикулам
    agg_cols = ["target"]
    if has_lower:
        agg_cols.append("target_0.1")
    if has_upper:
        agg_cols.append("target_0.95")

    forecast_summary = forecast_df.groupby(
        ["department", "article"], as_index=False
    )[agg_cols].sum()

    # округление квантилей для наглядности
    if has_lower:
        forecast_summary["target_0.1"] = forecast_summary["target_0.1"].round()
    if has_upper:
        forecast_summary["target_0.95"] = forecast_summary["target_0.95"].round()

    print("=== Прогноз по департаментам и артикулам ===")
    print(forecast_summary.sort_values(["department", "article"]).to_string(index=False))

    # печать сумм квантилей
    if has_lower or has_upper:
        lower_sum = forecast_df["target_0.1"].sum() if has_lower else 0
        upper_sum = forecast_df["target_0.95"].sum() if has_upper else 0
        print("=== Суммы квантилей ===")
        print(f"Σ target_0.1 (нижняя граница): {lower_sum:.2f}")
        print(f"Σ target_0.95 (верхняя граница): {upper_sum:.2f}")

    # формируем JSON по сегментам
    result = {}
    for seg in forecast.segments:
        cols = ["timestamp", "target"]
        if has_lower:
            cols.append("target_0.1")
        if has_upper:
            cols.append("target_0.95")

        seg_df = forecast_df[forecast_df["segment"] == seg][cols]

        result[seg] = [
            {
                "timestamp": row["timestamp"].strftime("%Y-%m-%d"),
                "target": max(0, float(row["target"])),
                "target_lower": max(0, float(row["target_0.1"])) if has_lower else None,
                "target_upper": max(0, float(row["target_0.95"])) if has_upper else None,
            }
        for _, row in seg_df.iterrows()
        ]

    json_str = json.dumps(result, ensure_ascii=False, indent=2)
    output_file = Path("forecast.json")
    output_file.write_text(json_str, encoding="utf-8")
    print(f"✅ JSON сохранён в {output_file.resolve()}")

    # график на основе forecast_df
    new_forecast_plot(forecast_df)
