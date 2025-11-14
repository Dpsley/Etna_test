import warnings
import pathlib
from etna.core import load
from etna.transforms import HolidayTransform, DateFlagsTransform, TimeSeriesImputerTransform, STLTransform, \
    LagTransform, SegmentEncoderTransform, LogTransform, OneHotEncoderTransform, LabelEncoderTransform

warnings.filterwarnings(action="ignore")

Transformers_saved_dir = pathlib.Path("saved_transformers")
Transformers_saved_dir.mkdir(parents=True, exist_ok=True)


def load_transformers() -> list|Exception:
    """
    Загружает трансформеры из папки и возвращает list[Transformers] или Exception.
    """
    try:
        transformers = []
        for file in sorted(Transformers_saved_dir.iterdir()):
            if file.suffix == ".zip" and file.name.startswith("transformer_"):
                transformer = load(file)
                transformers.append(**transformer.to_dict())
        return transformers
    except Exception as e:
        return e


def save_transformers(transformers: list) -> bool|Exception:
    """
    Сохраняем список трансформеров в папку.\n
    Каждый трансформер -> отдельный zip.\n
    Возвращает либо True, либо Exception.

    Parameters
    ----------
    transformers:
        Массив с трансформерами.
    """
    try:
        for i, transformer_element in enumerate(transformers):
            transformer_element.save(Transformers_saved_dir / f"transformer_{i}.zip")
        return True
    except Exception as e:
        return e

def transformers_generator() -> list:
    transforms = [
#        LagTransform(in_column="target", lags=[14, 21, 28, 30, 31], out_column="target_lag_small"),
#        LagTransform(in_column="target", lags=[1, 3, 5, 7], out_column="lags_macro"),
#        LagTransform(in_column="target", lags=[2, 4, 6], out_column="lags_macro_2"),
#        LogTransform(in_column="target", out_column="target_log", inplace=False),
        LogTransform(in_column="target"),
        HolidayTransform(out_column="holiday", iso_code="RUS"),
        DateFlagsTransform(
#            day_number_in_week=True,
#            day_number_in_week=False,
#            day_number_in_month=True,
#            day_number_in_year=True,
#            day_number_in_month=False,
#            day_number_in_year=False,
#            week_number_in_month=True,
#            week_number_in_year=True,
            month_number_in_year=True,
            season_number=True,
            year_number=True,
#            is_weekend=True,
            out_column="flag",
        ),
#        STLTransform(in_column="target", model="arima", period=91, robust=True, model_kwargs={"order": (2, 2, 1)}),
#        STLTransform(in_column="target", model="arima", period=3, robust=True, model_kwargs={"order": (2, 2, 1)}),
        TimeSeriesImputerTransform(in_column="target", strategy="seasonal"),
        SegmentEncoderTransform(),
        #OneHotEncoderTransform(in_column="Department"),
        #OneHotEncoderTransform(in_column="ProductGroup"),
        #OneHotEncoderTransform(in_column="ProductType"),
        #OneHotEncoderTransform(in_column="prop_Вид"),
        #OneHotEncoderTransform(in_column="prop_Регион"),
        #OneHotEncoderTransform(in_column="prop_Тип"),
        #OneHotEncoderTransform(in_column="prop_Марка"),
        #OneHotEncoderTransform(in_column="prop_Код производителя"),
        #OneHotEncoderTransform(in_column="prop_Вкус"),
        #OneHotEncoderTransform(in_column="prop_Ароматизированный")
    ]

    return transforms