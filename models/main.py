import warnings

from etna.models import CatBoostMultiSegmentModel, NonPredictionIntervalContextIgnorantAbstractModel, \
    NonPredictionIntervalContextRequiredAbstractModel, PredictionIntervalContextIgnorantAbstractModel, \
    PredictionIntervalContextRequiredAbstractModel

warnings.filterwarnings(action="ignore")
import pathlib
from etna.core import load


Model_saved_dir = pathlib.Path("saved_models")
Model_saved_dir.mkdir(parents=True, exist_ok=True)

def load_model_from_dump() -> CatBoostMultiSegmentModel|Exception:
    """
    Загружает модель из папки и возвращает экземпляр CatBoostMultiSegmentModel или Exception.
    """
    try:
        model = load(Model_saved_dir / "model.zip")
        return model
    except Exception as e:
        return e

def save_model(model: NonPredictionIntervalContextIgnorantAbstractModel | NonPredictionIntervalContextRequiredAbstractModel | PredictionIntervalContextIgnorantAbstractModel | PredictionIntervalContextRequiredAbstractModel) -> bool|Exception:
    """
    Сохраняет модель в папку и возвращает либо True, либо Exception.

    Parameters
    ----------
    model:
        CatBoostMultiSegmentModel.
    """
    try:
        model.save(Model_saved_dir / "model.zip")
        return True
    except Exception as e:
        return e
