import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from tabpfn import TabPFNRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error


def measure_random_forest_mean_absolute_error(
    synthetic_x: list[pd.DataFrame],
    synthetic_y: list[list[int]],
    reral_x: pd.DataFrame,
    real_y: list[int],
):
    mean_absolute_errors = []
    for synt_x, synth_y in zip(synthetic_x, synthetic_y):
        random_forest_regressor = RandomForestRegressor(
            n_estimators=len(reral_x) // 5,
        ).fit(synt_x, synth_y)
        mean_absolute_errors.append(
            mean_absolute_error(real_y, random_forest_regressor.predict(reral_x))
        )
    return mean_absolute_errors


def measure_tab_pfn_mean_absolute_error(
    synthetic_x: list[pd.DataFrame],
    synthetic_y: list[list[int]],
    reral_x: pd.DataFrame,
    real_y: list[int],
):
    mean_absolute_errors = []
    for synt_x, synth_y in zip(synthetic_x, synthetic_y):
        random_forest_regressor = TabPFNRegressor().fit(synt_x, synth_y)
        mean_absolute_errors.append(
            mean_absolute_error(real_y, random_forest_regressor.predict(reral_x))
        )
    return mean_absolute_errors


def measure_xgb_mean_absolute_error(
    synthetic_x: list[pd.DataFrame],
    synthetic_y: list[list[int]],
    reral_x: pd.DataFrame,
    real_y: list[int],
):
    mean_absolute_errors = []
    for synt_x, synth_y in zip(synthetic_x, synthetic_y):
        random_forest_regressor = XGBRegressor().fit(synt_x, synth_y)
        mean_absolute_errors.append(
            mean_absolute_error(real_y, random_forest_regressor.predict(reral_x))
        )
    return mean_absolute_errors
