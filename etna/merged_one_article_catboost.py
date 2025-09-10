seg = SegmentEncoderTransform()
lags = LagTransform(in_column="target", lags=[1, 2, 3, 7, 14, 21, 28, 30, 60, 90,180,210,240,270,300,360], out_column="target_lag")
stl = STLTransform(in_column="target", period=7)
trend = LinearTrendTransform(in_column="target")
imputer = TimeSeriesImputerTransform(in_column="target")
mean_tr3 = MeanTransform(in_column="target", out_column="mean_3", window=3)
mean_tr7 = MeanTransform(in_column="target", out_column="mean_7", window=7)
mean_tr14 = MeanTransform(in_column="target", out_column="mean_14", window=14)
mean_tr21 = MeanTransform(in_column="target", out_column="mean_21", window=21)
mean_tr28 = MeanTransform(in_column="target", out_column="mean_28", window=28)
mean_tr60 = MeanTransform(in_column="target", out_column="mean_60", window=60)
mean_tr90 = MeanTransform(in_column="target", out_column="mean_90", window=90)

std_7 = StdTransform(in_column="target", out_column="std_7", window=7)
std_30 = StdTransform(in_column="target", out_column="std_30", window=30)
#cp_seg = ChangePointsSegmentationTransform(in_column="target")
fourier = FourierTransform(in_column="target", out_column="fourier_target", period=365, order=1)
log_tr = LogTransform(in_column="target")  # создаем новую колонку target_log
robust = RobustScalerTransform(in_column="target", out_column="robust_target")
date_flags = DateFlagsTransform(
    day_number_in_week=True,
    day_number_in_month=True,
    day_number_in_year=True,
    week_number_in_month=True,
    week_number_in_year=True,
    month_number_in_year=True,
    year_number=True,
    is_weekend=True,
    out_column="flag",
)
holiday_tr = HolidayTransform(out_column="holiday", iso_code="RUS")

transforms = [
    log_tr,
    lags,
 #   mean_tr3,
 #   mean_tr7,
 #   mean_tr14,
 #   mean_tr21,
 #   mean_tr28,
 #   mean_tr60,
 #   mean_tr90,
 #   std_7,
#    std_30,
#    holiday_tr,
    date_flags,
    stl,
#    trend,
#    fourier,
#    robust,
    imputer,
    seg
]