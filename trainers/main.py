import os
from pathlib import Path
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

def fitter():
    train_ts = load_actual_dataset()
    best_trial = optuna_produce(train_ts)
    transformers=transformers_generator()
    final_model = CatBoostMultiSegmentModel(**best_trial.params, logging_level="Silent")
    pipeline = Pipeline(model=final_model, transforms=transformers, horizon=HORIZON)
    pipeline.fit(ts=train_ts)
    save_pipline(pipeline)
    save_model(pipeline.model)
    save_transformers(transformers)

def optuna_produce(train_ts) -> FrozenTrial:
    def objective(trial):
        best_params = {
            'iterations': trial.suggest_int('iterations', 500, 3000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.6, log=True),
            'depth': trial.suggest_int('depth', 4, 16),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.01, 10.0, log=True),
            'random_strength': trial.suggest_float('random_strength', 0.1, 20.0),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 5.0),
            'border_count': trial.suggest_int('border_count', 32, 512),
            'one_hot_max_size': trial.suggest_int('one_hot_max_size', 2, 25),
            'loss_function': 'RMSE',
            'early_stopping_rounds': 200,
            'random_seed': 42
        }
        model = CatBoostMultiSegmentModel(**best_params)
        pipeline = Pipeline(model=model, transforms=transformers_generator(), horizon=123)
        pipeline.fit(ts=train_ts)
        backtest_result = pipeline.backtest(
            ts=train_ts,
            metrics=[RMSE()],
            n_folds=1,
            mode="constant"
        )
        mean_rmse = backtest_result["metrics"]["RMSE"].mean().mean()
        return mean_rmse

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=1, show_progress_bar=True)
    return study.best_trial