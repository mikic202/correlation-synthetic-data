import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


def measure_random_forest_mean_absolute_error(
    synthetic_x: list[pd.DataFrame],
    synthetic_y: list[list[int]],
    reral_x: pd.DataFrame,
    real_y: list[int],
):
    mean_absolute_errors = []
    for synt_x, synth_y in zip(synthetic_x, synthetic_y):
        random_forest_regressor = RandomForestRegressor().fit(synt_x, synth_y)
        mean_absolute_errors.append(
            mean_absolute_error(real_y, random_forest_regressor.predict(reral_x))
        )
    return mean_absolute_errors
