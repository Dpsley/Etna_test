(venv) supervisor@master-llm-alephtrade:~/Etna_test$ cat trainers/main.py
import os
from etna.ensembles import VotingEnsemble
from etna.models import CatBoostMultiSegmentModel, CatBoostPerSegmentModel, ProphetModel, SeasonalMovingAverageModel, \
    NaiveModel
from etna.metrics import RMSE, MAE, MSE, SMAPE, MAPE
from etna.pipeline import Pipeline
import optuna
from optuna.trial import FrozenTrial
from datasets.main import load_actual_dataset
from models.main import save_model
from pipelines.main import save_pipline
from etna_transformers.main import transformers_generator, save_transformers
from dotenv import load_dotenv
import pandas as pd
from etna.datasets import TSDataset

load_dotenv()

HORIZON = os.getenv("HORIZON")
print(int(HORIZON))

def fitter():
    print(int(HORIZON))
    print("Запуск fitter")
    train_ts = load_actual_dataset()
    transformers=transformers_generator()
    print(train_ts)
    best_trial = optuna_produce(train_ts, transformers)
    print(best_trial)
    #best_trial.params["border_count"] = 32
    final_model = CatBoostMultiSegmentModel(**best_trial.params, logging_level="Silent", task_type="GPU", gpu_cat_features_storage="CpuPinnedMemory",  leaf_estimation_iterations=3, max_ctr_complexity=1, boosting_type="Plain")
    pipeline = Pipeline(model=final_model, transforms=transformers, horizon=int(HORIZON))
    pipeline.fit(ts=train_ts)
    #forecast= pipeline.forecast(prediction_interval=True)
    #forecast_df = forecast.to_pandas(flatten=True).reset_index()
    #forecast_df["timestamp"] = pd.to_datetime(forecast_df["timestamp"])
    #forecast_df["department"] = forecast_df["segment"].str.split("|").str[0]
    #forecast_df["article"] = forecast_df["segment"].str.split("|").str[1]
    #forecast_agg = forecast_df.groupby(["timestamp", "department", "article"], as_index=False)["target"].sum()
    #print("=== Прогноз по департаментам и артикулам ===")
    #forecast_summary = forecast_agg.groupby(["department", "article"], as_index=False)["target"].sum()
    #print(forecast_summary.sort_values(["department", "article"]).to_string(index=False))
    save_pipline(pipeline)
    save_model(pipeline.model)
    save_transformers(transformers)

def optuna_produce(train_ts, transformers) -> FrozenTrial:
    def objective(trial):
#        return 1000
        best_params = {
            'iterations': trial.suggest_int('iterations', 500, 1500),
            'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.8, log=True),
            'depth': trial.suggest_int('depth', 4, 8),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.1, 20, log=True),
            'random_strength': trial.suggest_float('random_strength', 0.01, 20.0),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 5.0),
            'border_count': trial.suggest_int('border_count', 8, 512),
            'one_hot_max_size': trial.suggest_int('one_hot_max_size', 2, 25),
            'loss_function': 'RMSE',
            'early_stopping_rounds': 100,
            'random_seed': 42,
        }
        model = CatBoostMultiSegmentModel(**best_params)
        pipeline = Pipeline(model=model, transforms=transformers, horizon=int(HORIZON))
        pipeline.fit(ts=train_ts)
        backtest_result = pipeline.backtest(
            ts=train_ts,
            metrics=[RMSE()],
            n_folds=5,
            mode="constant"
        )
        # по каждому фолду репортим промежуточный результат
        fold_scores = backtest_result["metrics"]["RMSE"].values
        print("RMSE:", backtest_result["metrics"]["RMSE"].mean().mean())
        for step, score in enumerate(fold_scores):
            trial.report(score, step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        mean_rmse = backtest_result["metrics"]["RMSE"].mean().mean()
        return mean_rmse

    study = optuna.create_study(direction="minimize" ,
                                #storage="sqlite:///db.sqlite3",
                                #study_name="monthly_test_4",
                                #load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=42),
                                pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
                                )
    study.optimize(objective, n_trials=100, show_progress_bar=True)
    return study.best_trial

def fitter_ensemble():
    import json
    from pathlib import Path

    print(int(HORIZON))
    print("запуск ансамбля")
    train_ts = load_actual_dataset()
    transformers=transformers_generator()
    print(train_ts)
#    best_trial = optuna_produce(train_ts, transformers)
#    print(best_trial)

    naive_pipeline = Pipeline(
        model=NaiveModel(lag=12),
        transforms=transformers,
        horizon=int(HORIZON)
    )
    seasonalma_pipeline = Pipeline(
        model=SeasonalMovingAverageModel(window=2, seasonality=3),
        transforms=transformers,
        horizon=int(HORIZON),
    )
    catboost_pipeline = Pipeline(
        model=CatBoostMultiSegmentModel(),
        transforms=transformers,
        horizon=int(HORIZON),
    )

    pipelines = [naive_pipeline, seasonalma_pipeline, catboost_pipeline]
    voting_ensemble = VotingEnsemble(pipelines=pipelines, weights=[1,9,4], n_jobs=4)
    backtest_result = voting_ensemble.backtest(
        ts=train_ts,
        metrics=[MAE(), MSE(), SMAPE(), MAPE()],
        n_folds=3,
        aggregate_metrics=True,
        n_jobs=1,
    )
    # усредняем по всем сегментам и фолдам, чтобы был один итог
    voting_ensamble_metrics = backtest_result["metrics"].iloc[:, 1:].mean().to_frame().T
    voting_ensamble_metrics.index = ["voting ensemble"]
    forecast = voting_ensemble.forecast(prediction_interval=True, quantiles=[0.025, 0.975])
    print("forecast", forecast)

    forecast_df = forecast.to_pandas(flatten=True).reset_index()
    forecast_df["timestamp"] = pd.to_datetime(forecast_df["timestamp"])
    forecast_df["department"] = forecast_df["segment"].str.split("|").str[0]
    forecast_df["article"] = forecast_df["segment"].str.split("|").str[1]

    # проверим, есть ли колонки квантилей
    has_lower = "target_0.025" in forecast_df.columns
    has_upper = "target_0.975" in forecast_df.columns

    # создаём колонку для квантилей, обрезаем отрицательные значения
    if has_lower:
        forecast_df["target_0.025"] = forecast_df["target_0.025"].apply(lambda x: max(float(x), 0))
    if has_upper:
        forecast_df["target_0.975"] = forecast_df["target_0.975"].apply(lambda x: max(float(x), 0))

    # агрегация по департаментам и артикулам
    agg_cols = ["target"]
    if has_lower:
        agg_cols.append("target_0.025")
    if has_upper:
        agg_cols.append("target_0.975")

    forecast_summary = forecast_df.groupby(
        ["department", "article"], as_index=False
    )[agg_cols].sum()

    # округление квантилей для наглядности
    if has_lower:
        forecast_summary["target_0.025"] = forecast_summary["target_0.025"].round()
    if has_upper:
        forecast_summary["target_0.975"] = forecast_summary["target_0.975"].round()

    print("=== Прогноз по департаментам и артикулам ===")
    print(forecast_summary.sort_values(["department", "article"]).to_string(index=False))

    # печать сумм квантилей
    if has_lower or has_upper:
        lower_sum = forecast_df["target_0.025"].sum() if has_lower else 0
        upper_sum = forecast_df["target_0.975"].sum() if has_upper else 0
        print("=== Суммы квантилей ===")
        print(f"Σ target_0.025 (нижняя граница): {lower_sum:.2f}")
        print(f"Σ target_0.975 (верхняя граница): {upper_sum:.2f}")

    # формируем JSON по сегментам
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
                "target": row["target"].apply(lambda x: max(0, float(x))),
                "target_lower": row["target_0.025"].apply(lambda x: max(0, float(x))) if has_lower else None,
                "target_upper": row["target_0.975"].apply(lambda x: max(0, float(x))) if has_upper else None,
            }
            for _, row in seg_df.iterrows()
        ]

    json_str = json.dumps(result, ensure_ascii=False, indent=2)
    output_file = Path("forecast.json")
    output_file.write_text(json_str, encoding="utf-8")
    print(f"✅ JSON сохранён в {output_file.resolve()}")

fitter_ensemble()