from test_datasets.dataset_getters import (
    get_pc4_dataset,
    get_mfeat_zernike_dataset,
    get_climate_model_simulation_dataset,
    get_wdbc_dataset,
    get_analcatdata_authorship_dataset,
)
from performance_metrics.measure_logistic_regresion_auc import (
    measure_logistic_regresion_auc,
)
from performance_metrics.measure_random_forest_auc import measure_random_forest_auc
from performance_metrics.measure_xgb_auc import measure_xgb_auc
from performance_metrics.measure_tabpfn_auc import measure_tabpfn_auc

import pandas as pd


AVAILABLE_DATASETS = {
    "mfeat_zernike": get_mfeat_zernike_dataset,
    "pc4": get_pc4_dataset,
    "climate_model_simulation": get_climate_model_simulation_dataset,
    "wdbc": get_wdbc_dataset,
    "analcatdata_authorship": get_analcatdata_authorship_dataset,
}


def measure_model_performance(model, **kwargs):
    results = pd.DataFrame(
        columns=["dataset", "random_forest", "xgboost", "LR", "TabPFN"]
    )
    for dataset_name, dataset_getter in AVAILABLE_DATASETS.items():
        train, _ = dataset_getter()
        real_x, real_y = train.drop("c", axis=1), train["c"].to_list()
        synth_x, synth_y = model(
            real_x,
            real_y,
            n_samples=real_x.shape[0],
            balance_classes=True,
        )
        results.loc[-1] = [dataset_name, 0.0, 0.0, 0.0, 0.0]
        results.loc[-1, "random_forest"] = measure_random_forest_auc(
            [synth_x], [synth_y], real_x, real_y
        )
        results.loc[-1, "xgboost"] = measure_xgb_auc(
            [synth_x], [synth_y], real_x, real_y
        )
        results.loc[-1, "LR"] = measure_logistic_regresion_auc(
            [synth_x], [synth_y], real_x, real_y
        )
        results.loc[-1, "TabPFN"] = measure_tabpfn_auc(
            [synth_x], [synth_y], real_x, real_y
        )
        results.index = results.index + 1
    return results
