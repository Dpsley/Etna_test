from etna.pipeline import Pipeline
from datasets.main import load_actual_dataset


def new_forecast(pipeline: Pipeline):
    pipeline.forecast(prediction_interval=True, ts=load_actual_dataset())