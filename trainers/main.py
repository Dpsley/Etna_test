import os
from etna.ensembles import VotingEnsemble, StackingEnsemble
from etna.models import CatBoostMultiSegmentModel, CatBoostPerSegmentModel, ProphetModel, SeasonalMovingAverageModel, \
    NaiveModel, SklearnMultiSegmentModel
from etna.metrics import RMSE, MAE, MSE, SMAPE, MAPE
from etna.pipeline import Pipeline
import optuna
from optuna.trial import FrozenTrial
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet, SGDRegressor, BayesianRidge, LinearRegression
from sklearn.svm import SVR

from datasets.main import load_actual_dataset
from models.main import save_model
from pipelines.main import save_pipline
from etna_transformers.main import transformers_generator, save_transformers
from dotenv import load_dotenv
import pandas as pd
from etna.datasets import TSDataset
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

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
    import numpy as np

    print(int(HORIZON))
    print("запуск ансамбля")
    train_ts = load_actual_dataset()
    transformers = transformers_generator()
    print(train_ts)

    # --- фабрики пайплайнов, чтобы можно было создать свежий экземпляр ---
    def make_naive_1():
        return Pipeline(
            model=NaiveModel(lag=1),
            transforms=transformers,
            horizon=int(HORIZON),
        )

    def make_naive_3():
        return Pipeline(
            model=NaiveModel(lag=3),
            transforms=transformers,
            horizon=int(HORIZON),
        )

    def make_naive_6():
        return Pipeline(
            model=NaiveModel(lag=6),
            transforms=transformers,
            horizon=int(HORIZON),
        )

    def make_naive_12():
        return Pipeline(
            model=NaiveModel(lag=12),
            transforms=transformers,
            horizon=int(HORIZON),
        )

    def make_seasonal_1_3():
        return Pipeline(
            model=SeasonalMovingAverageModel(window=1, seasonality=3),
            transforms=transformers,
            horizon=int(HORIZON),
        )

    def make_seasonal_1_6():
        return Pipeline(
            model=SeasonalMovingAverageModel(window=1, seasonality=6),
            transforms=transformers,
            horizon=int(HORIZON),
        )

    def make_seasonal_2_3():
        # у тебя тут был seasonality=1 с названием _2_3 — это точно баг
        return Pipeline(
            model=SeasonalMovingAverageModel(window=2, seasonality=3),
            transforms=transformers,
            horizon=int(HORIZON),
        )

    def make_seasonal_2_6():
        return Pipeline(
            model=SeasonalMovingAverageModel(window=2, seasonality=6),
            transforms=transformers,
            horizon=int(HORIZON),
        )

    def make_catboost():
        # можно сюда потом подставить best_params из optuna, пока — дефолт
        return Pipeline(
            model=CatBoostMultiSegmentModel(),
            transforms=transformers,
            horizon=int(HORIZON),
        )

    def make_xgboost():
        return Pipeline(
            model=SklearnMultiSegmentModel(regressor=XGBRegressor()),
            transforms=transformers,
            horizon=int(HORIZON),
        )

    def make_lightgbm():
        return Pipeline(
            model=SklearnMultiSegmentModel(regressor=LGBMRegressor()),
            transforms=transformers,
            horizon=int(HORIZON),
        )

    # --- словарь всех базовых моделей ---
    pipeline_factories = {
        "naive_1": make_naive_1,
        "naive_3": make_naive_3,
        "naive_6": make_naive_6,
        "naive_12": make_naive_12,
        "seasonal_1_3": make_seasonal_1_3,
        "seasonal_1_6": make_seasonal_1_6,
        "seasonal_2_3": make_seasonal_2_3,
        "seasonal_2_6": make_seasonal_2_6,
        "catboost": make_catboost,
        "xgboost": make_xgboost,
        "lightgbm": make_lightgbm,
    }

    # --- оцениваем каждую модель по MAE на backtest ---
    model_mae = {}

    for name, factory in pipeline_factories.items():
        print(f"\n=== Оценка модели {name} ===")
        pl = factory()
        bt_res = pl.backtest(
            ts=train_ts,
            metrics=[MAE()],
            n_folds=3,
            aggregate_metrics=True,
            n_jobs=1,
        )
        # MAE может быть DataFrame, берём среднее по фолдам
        mae_val = float(bt_res["metrics"]["MAE"].mean())
        model_mae[name] = mae_val
        print(f"{name} -> MAE = {mae_val:.4f}")

    # --- выбираем топ-N моделей по MAE ---
    TOP_N = 5  # можешь поиграть: 3–6
    sorted_models = sorted(model_mae.items(), key=lambda x: x[1])
    top_models = sorted_models[:TOP_N]
    print("\n=== Топ моделей по MAE ===")
    for name, mae_val in top_models:
        print(f"{name}: MAE={mae_val:.4f}")

    # --- считаем веса: чем меньше MAE, тем больше вес ---
    inv_mae = np.array([1.0 / mae for _, mae in top_models])
    inv_mae = inv_mae / inv_mae.sum()
    weights = inv_mae.tolist()

    selected_names = [name for name, _ in top_models]
    print("\n=== Ансамбль ===")
    print("Модели:", selected_names)
    print("Веса:", [round(w, 4) for w in weights])

    # --- создаём новые экземпляры пайплайнов для финального ансамбля ---
    selected_pipelines = [pipeline_factories[name]() for name in selected_names]

    voting_ensemble = VotingEnsemble(
        pipelines=selected_pipelines,
        weights=weights,  # вместо CatBoostRegressor как мета-модели
    )

    # ===== Финальный backtest ансамбля =====
    backtest_result = voting_ensemble.backtest(
        ts=train_ts,
        metrics=[MAE(), MSE(), SMAPE(), MAPE()],
        n_folds=3,
        aggregate_metrics=True,
        n_jobs=1,
    )

    voting_ensamble_metrics = backtest_result["metrics"].iloc[:, 1:].mean().to_frame().T
    voting_ensamble_metrics.index = ["voting ensemble"]
    print("\n=== Метрики ансамбля ===")
    print(voting_ensamble_metrics.round(4))

    # ===== Обучаем на всех данных и строим прогноз =====
    voting_ensemble.fit(ts=train_ts)
    forecast = voting_ensemble.forecast(prediction_interval=True, quantiles=[0.025, 0.975])
    print("forecast", forecast)

    # === метаданные по сегментам ===
    meta_df = (
        train_ts.to_pandas(flatten=True)
        .reset_index()
        .sort_values("timestamp")
        .groupby("segment", as_index=False)[["ProductType", "ProductName"]]
        .last()
        .rename(columns={"ProductType": "department", "ProductName": "article"})
    )

    forecast_df = forecast.to_pandas(flatten=True).reset_index()
    forecast_df["timestamp"] = pd.to_datetime(forecast_df["timestamp"])

    forecast_df = forecast_df.merge(meta_df, on="segment", how="left")

    has_lower = "target_0.025" in forecast_df.columns
    has_upper = "target_0.975" in forecast_df.columns

    if has_lower:
        forecast_df["target_0.025"] = forecast_df["target_0.025"].apply(lambda x: max(float(x), 0))
    if has_upper:
        forecast_df["target_0.975"] = forecast_df["target_0.975"].apply(lambda x: max(float(x), 0))

    forecast_df["department"] = forecast_df["department"].fillna("UNKNOWN")
    forecast_df["article_code"] = forecast_df["segment"]
    forecast_df["article"] = forecast_df["article"].fillna(forecast_df["article_code"])

    agg_cols = ["target"]
    if has_lower:
        agg_cols.append("target_0.025")
    if has_upper:
        agg_cols.append("target_0.975")

    forecast_summary = (
        forecast_df.groupby(["department", "article"], as_index=False)[agg_cols]
        .sum()
    )

    if has_lower:
        forecast_summary["target_0.025"] = forecast_summary["target_0.025"].round()
    if has_upper:
        forecast_summary["target_0.975"] = forecast_summary["target_0.975"].round()

    print("=== Прогноз по департаментам и артикулам ===")
    if not forecast_summary.empty:
        print(forecast_summary.sort_values(["department", "article"]).to_string(index=False))
    else:
        print("Пусто — проверь метаданные сегментов.")

    if has_lower or has_upper:
        lower_sum = forecast_df["target_0.025"].sum() if has_lower else 0
        upper_sum = forecast_df["target_0.975"].sum() if has_upper else 0
        print("=== Суммы квантилей ===")
        print(f"Σ target_0.025 (нижняя граница): {lower_sum:.2f}")
        print(f"Σ target_0.975 (верхняя граница): {upper_sum:.2f}")

    # === JSON по сегментам ===
    result = {}
    for seg, seg_df in forecast_df.groupby("segment"):
        rows = []
        for _, r in seg_df.iterrows():
            item = {
                "timestamp": r["timestamp"].strftime("%Y-%m-%d"),
                "target": max(0.0, float(r["target"])),
            }
            if has_lower:
                item["target_lower"] = max(0.0, float(r["target_0.025"]))
            if has_upper:
                item["target_upper"] = max(0.0, float(r["target_0.975"]))
            rows.append(item)
        result[seg] = rows

    json_str = json.dumps(result, ensure_ascii=False, indent=2)
    output_file = Path("forecast.json")
    output_file.write_text(json_str, encoding="utf-8")
    print(f"✅ JSON сохранён в {output_file.resolve()}")



fitter_ensemble()
