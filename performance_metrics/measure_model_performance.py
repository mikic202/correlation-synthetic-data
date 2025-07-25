from test_datasets.dataset_getters import (
    get_pc4_dataset,
    get_mfeat_zernike_dataset,
    get_climate_model_simulation_dataset,
    get_wdbc_dataset,
    get_analcatdata_authorship_dataset,
    get_heart_failure_clinical_regresion_dataset,
    get_sleep_deprivation_and_cognitive_performance_regression_dataset,
    get_superconduct_regression_dataset,
    get_house_prices_regression_dataset,
    REGRESION_TARGET,
    CLASYFICATION_TARGET,
)
from performance_metrics.measure_area_under_curve import (
    measure_logistic_regresion_auc,
    measure_random_forest_auc,
    measure_xgb_auc,
    measure_tabpfn_auc,
)
from performance_metrics.measure_mean_absolute_error import (
    measure_linear_regresion_mean_absolute_error,
    measure_random_forest_mean_absolute_error,
    measure_xgb_mean_absolute_error,
    measure_tab_pfn_mean_absolute_error,
)
import pandas as pd


AVAILABLE_DATASETS = {
    "mfeat_zernike": get_mfeat_zernike_dataset,
    "pc4": get_pc4_dataset,
    "climate_model_simulation": get_climate_model_simulation_dataset,
    "wdbc": get_wdbc_dataset,
    "analcatdata_authorship": get_analcatdata_authorship_dataset,
}


AVALIABLE_REGRESSION_DATASETS = {
    "heart_failure_clinical_regresion": get_heart_failure_clinical_regresion_dataset,
    "sleep_deprivation_and_cognitive_performance_regression": get_sleep_deprivation_and_cognitive_performance_regression_dataset,
    "superconduct_regression": get_superconduct_regression_dataset,
    "house_prices_regression": get_house_prices_regression_dataset,
}


RANDOM_FOREST_COLUMN = "random_forest"
DATASET_COLUMN = "dataset"
XGBOOST_COLUMN = "xgboost"
TABPFN_COLUMN = "TabPFN"
LOGISTIC_REGRESION_COLUMN = "LR"
LINEAR_REGRESION_COLUMN = "linear_regression"


def measure_model_performance(model, **kwargs):
    results = pd.DataFrame(
        columns=[
            DATASET_COLUMN,
            RANDOM_FOREST_COLUMN,
            XGBOOST_COLUMN,
            LOGISTIC_REGRESION_COLUMN,
            TABPFN_COLUMN,
        ]
    )
    for dataset_name, dataset_getter in AVAILABLE_DATASETS.items():
        train, test = dataset_getter()
        real_x, real_y = (
            train.drop(CLASYFICATION_TARGET, axis=1),
            train[CLASYFICATION_TARGET].to_list(),
        )
        test_x, test_y = (
            test.drop(CLASYFICATION_TARGET, axis=1),
            test[CLASYFICATION_TARGET].to_list(),
        )
        synth_x, synth_y = model(
            real_x,
            real_y,
            n_samples=real_x.shape[0],
            balance_classes=True,
            **kwargs,
        )
        synth_x = pd.DataFrame(synth_x, columns=real_x.columns)
        results.loc[-1] = [dataset_name, 0.0, 0.0, 0.0, 0.0]
        results.loc[-1, RANDOM_FOREST_COLUMN] = measure_random_forest_auc(
            [synth_x], [synth_y], test_x, test_y
        )
        results.loc[-1, XGBOOST_COLUMN] = measure_xgb_auc(
            [synth_x], [synth_y], test_x, test_y
        )
        results.loc[-1, LOGISTIC_REGRESION_COLUMN] = measure_logistic_regresion_auc(
            [synth_x], [synth_y], test_x, test_y
        )
        results.loc[-1, TABPFN_COLUMN] = measure_tabpfn_auc(
            [synth_x], [synth_y], test_x, test_y
        )
        results.index = results.index + 1
    return results


def measure_regresion_model_performance(model, **kwargs):
    results = pd.DataFrame(
        columns=[
            DATASET_COLUMN,
            RANDOM_FOREST_COLUMN,
            XGBOOST_COLUMN,
            TABPFN_COLUMN,
            LINEAR_REGRESION_COLUMN,
        ]
    )
    for dataset_name, dataset_getter in AVALIABLE_REGRESSION_DATASETS.items():
        train, test = dataset_getter()
        train = train[: min(10000, len(train))]  # CUDA out of memory error prevention
        real_x, real_y = (
            train.drop(REGRESION_TARGET, axis=1),
            train[REGRESION_TARGET].to_numpy(),
        )
        synth_x, synth_y = model(
            real_x,
            real_y,
            n_samples=real_x.shape[0],
            **kwargs,
        )
        real_x, real_y = (
            test.drop(REGRESION_TARGET, axis=1),
            test[REGRESION_TARGET].to_numpy(),
        )
        results.loc[-1] = [dataset_name, 0.0, 0.0, 0.0, 0.0]
        results.loc[-1, RANDOM_FOREST_COLUMN] = (
            measure_random_forest_mean_absolute_error(
                [synth_x], [synth_y], real_x, real_y
            )
        )
        results.loc[-1, XGBOOST_COLUMN] = measure_xgb_mean_absolute_error(
            [synth_x], [synth_y], real_x, real_y
        )
        results.loc[-1, TABPFN_COLUMN] = measure_tab_pfn_mean_absolute_error(
            [synth_x], [synth_y], real_x, real_y
        )
        results.loc[-1, LINEAR_REGRESION_COLUMN] = (
            measure_linear_regresion_mean_absolute_error(
                [synth_x], [synth_y], real_x, real_y
            )
        )
        results.index = results.index + 1
    return results
