import os
import pandas as pd
from etna.models import CatBoostMultiSegmentModel
from etna.metrics import RMSE
from etna.pipeline import Pipeline
import optuna
from optuna.trial import FrozenTrial
from datasets.main import load_actual_dataset
from models.main import save_model
from pipelines.main import save_pipline
from etna_transformers.main import transformers_generator, save_transformers
from dotenv import load_dotenv

load_dotenv()

HORIZON = os.getenv("HORIZON")
print(int(HORIZON))

def fitter():
    print(int(HORIZON))
    train_ts = load_actual_dataset()
    transformers=transformers_generator()
    print(train_ts)
    best_trial = optuna_produce(train_ts, transformers)
    print(best_trial)
    final_model = CatBoostMultiSegmentModel(**best_trial.params, logging_level="Silent")
    pipeline = Pipeline(model=final_model, transforms=transformers, horizon=int(HORIZON))
    pipeline.fit(ts=train_ts)
    forecast= pipeline.forecast(prediction_interval=True)
    forecast_df = forecast.to_pandas(flatten=True).reset_index()
    forecast_df["timestamp"] = pd.to_datetime(forecast_df["timestamp"])
    forecast_df["department"] = forecast_df["segment"].str.split("|").str[0]
    forecast_df["article"] = forecast_df["segment"].str.split("|").str[1]
    forecast_agg = forecast_df.groupby(["timestamp", "department", "article"], as_index=False)["target"].sum()
    print("=== Прогноз по департаментам и артикулам ===")
    forecast_summary = forecast_agg.groupby(["department", "article"], as_index=False)["target"].sum()
    print(forecast_summary.sort_values(["department", "article"]).to_string(index=False))
    save_pipline(pipeline)
    save_model(pipeline.model)
    save_transformers(transformers)

def optuna_produce(train_ts, transformers) -> FrozenTrial:
    def objective(trial):
        best_params = {
            'iterations': trial.suggest_int('iterations', 300, 2500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.7, log=True),
            #'iterations': trial.suggest_int('iterations', 1196, 1196),
            #'learning_rate': trial.suggest_float('learning_rate', 0.2928774987461045, 0.2928774987461045, log=True),
            'depth': trial.suggest_int('depth', 6, 14),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.001, 0.3, log=True),
            'random_strength': trial.suggest_float('random_strength', 0.4, 6),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 1, 5),
            'border_count': trial.suggest_int('border_count', 16, 350),
            'one_hot_max_size': trial.suggest_int('one_hot_max_size', 2, 30),
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
            mode="expand"
        )
        # по каждому фолду репортим промежуточный результат
        fold_scores = backtest_result["metrics"]["RMSE"].values
        for step, score in enumerate(fold_scores):
            trial.report(score, step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        mean_rmse = backtest_result["metrics"]["RMSE"].mean().mean()
        return mean_rmse

    study = optuna.create_study(direction="minimize" ,
                                storage="sqlite:///db.sqlite3",
                                study_name="quadratic-simple_prefinal",
                                load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=42),
                                pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
                                )
    study.optimize(objective, n_trials=100, show_progress_bar=True)
    return study.best_trial

fitter()