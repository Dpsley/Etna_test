import warnings

from datasets.main import load_actual_dataset

warnings.filterwarnings(action="ignore")
import pathlib
from etna.core import load
from etna.pipeline import Pipeline


Pipline_saved_dir = pathlib.Path("saved_pipelines")
Pipline_saved_dir.mkdir(parents=True, exist_ok=True)

def load_pipline_from_dump() -> Pipeline|Exception:
    """
    Загружает пайплайн из папки и возвращает Pipeline или Exception.
    """
    try:
        pipeline = load(Pipline_saved_dir / "pipeline.zip", ts=load_actual_dataset())
        print("pipeline loaded", pipeline)
        return pipeline
    except Exception as e:
        return e

def save_pipline(pipeline: Pipeline) -> bool|Exception:
    """
    Сохраняет пайплайн в папку и возвращает либо True, либо Exception.

    Parameters
    ----------
    pipeline:
        Pipeline.
    """
    try:
        pipeline.save(Pipline_saved_dir / "pipeline.zip")
        return True
    except Exception as e:
        return e