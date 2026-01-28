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
    get_cardiovascular_dataset,
    REGRESION_TARGET,
    CLASYFICATION_TARGET,
)
from performance_metrics.measure_area_under_curve import (
    measure_logistic_regresion_auc,
    measure_random_forest_auc,
    measure_xgb_auc,
    measure_tabpfn_auc,
    measure_tabicl_auc,
)
from performance_metrics.measure_mean_absolute_error import (
    measure_linear_regresion_mean_absolute_error,
    measure_random_forest_mean_absolute_error,
    measure_xgb_mean_absolute_error,
    measure_tab_pfn_mean_absolute_error,
)

from performance_metrics.measure_privacy import (
    calculate_k_anonimity_for_datset,
    calculate_distance_to_nearest_neighbour,
)
from performance_metrics.measure_synthetic_data_coverage import (
    calculate_synthetic_data_coverage,
)
from performance_metrics.measure_dataset_statistics import (
    measure_dataset_statistics,
    measure_matrix_statistics,
)
import pandas as pd
import numpy as np
import multiprocessing as mp
from multiprocessing import Process, Queue


AVAILABLE_DATASETS = {
    "mfeat_zernike": get_mfeat_zernike_dataset,
    "pc4": get_pc4_dataset,
    "climate_model_simulation": get_climate_model_simulation_dataset,
    "wdbc": get_wdbc_dataset,
    "analcatdata_authorship": get_analcatdata_authorship_dataset,
    "cardiovascular": get_cardiovascular_dataset,
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
TABICL_COLUMN = "TabICL"
LINEAR_REGRESION_COLUMN = "linear_regression"


def random_forest_process(
    downstream_results_queue: Queue,
    synth_x: pd.DataFrame,
    synth_y: pd.DataFrame,
    test_x: pd.DataFrame,
    test_y: pd.DataFrame,
):
    downstream_results_queue.put(
        (0, measure_random_forest_auc([synth_x], [synth_y], test_x, test_y)[0])
    )


def xgb_process(
    downstream_results_queue: Queue,
    synth_x: pd.DataFrame,
    synth_y: pd.DataFrame,
    test_x: pd.DataFrame,
    test_y: pd.DataFrame,
):
    downstream_results_queue.put(
        (1, measure_xgb_auc([synth_x], [synth_y], test_x, test_y)[0])
    )


def tabpfn_process(
    downstream_results_queue: Queue,
    synth_x: pd.DataFrame,
    synth_y: pd.DataFrame,
    test_x: pd.DataFrame,
    test_y: pd.DataFrame,
):
    downstream_results_queue.put(
        (3, measure_tabpfn_auc([synth_x], [synth_y], test_x, test_y)[0])
    )


def logistic_regression_process(
    downstream_results_queue: Queue,
    synth_x: pd.DataFrame,
    synth_y: pd.DataFrame,
    test_x: pd.DataFrame,
    test_y: pd.DataFrame,
):
    downstream_results_queue.put(
        (2, measure_logistic_regresion_auc([synth_x], [synth_y], test_x, test_y)[0])
    )


def tabicl_process(
    downstream_results_queue: Queue,
    synth_x: pd.DataFrame,
    synth_y: pd.DataFrame,
    test_x: pd.DataFrame,
    test_y: pd.DataFrame,
):
    downstream_results_queue.put(
        (4, measure_tabicl_auc([synth_x], [synth_y], test_x, test_y)[0])
    )


def measure_model_clasification_performance_once(
    model,
    real_x: pd.DataFrame,
    real_y: pd.DataFrame,
    test_x: pd.DataFrame,
    test_y: pd.DataFrame,
    n_samples: int | None = None,
    **kwargs,
) -> list[float]:
    mp.set_start_method("spawn", force=True)
    synth_x, synth_y = model(
        real_x,
        real_y,
        n_samples=n_samples if n_samples else real_x.shape[0],
        balance_classes=True,
        **kwargs,
    )
    synth_x = pd.DataFrame(synth_x, columns=real_x.columns)
    downstream_results_queue = Queue()
    downstream_jobs = []

    downstream_jobs.append(
        Process(
            target=random_forest_process,
            args=(downstream_results_queue, synth_x, synth_y, test_x, test_y),
        )
    )

    downstream_jobs.append(
        Process(
            target=xgb_process,
            args=(downstream_results_queue, synth_x, synth_y, test_x, test_y),
        )
    )
    downstream_jobs.append(
        Process(
            target=tabpfn_process,
            args=(downstream_results_queue, synth_x, synth_y, test_x, test_y),
        )
    )
    downstream_jobs.append(
        Process(
            target=logistic_regression_process,
            args=(downstream_results_queue, synth_x, synth_y, test_x, test_y),
        )
    )
    downstream_jobs.append(
        Process(
            target=tabicl_process,
            args=(downstream_results_queue, synth_x, synth_y, test_x, test_y),
        )
    )

    for job in downstream_jobs:
        job.start()
    for job in downstream_jobs:
        print(f"Waiting for {job.name} to finish...")
        job.join()
    return [
        accuracy
        for _, accuracy in sorted(
            downstream_results_queue.get() for _ in range(len(downstream_jobs))
        )
    ]


def measure_model_clasification_performance(
    model, number_of_repetitions: int = 5, n_samples: int | None = None, **kwargs
):
    results = pd.DataFrame(
        columns=[
            DATASET_COLUMN,
            RANDOM_FOREST_COLUMN,
            XGBOOST_COLUMN,
            LOGISTIC_REGRESION_COLUMN,
            TABPFN_COLUMN,
            TABICL_COLUMN,
        ]
    )
    for dataset_name, dataset_getter in AVAILABLE_DATASETS.items():
        train, test = dataset_getter()
        downstream_accuracies = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        real_x, real_y = (
            train.drop(CLASYFICATION_TARGET, axis=1),
            train[CLASYFICATION_TARGET].to_list(),
        )
        test_x, test_y = (
            test.drop(CLASYFICATION_TARGET, axis=1),
            test[CLASYFICATION_TARGET].to_list(),
        )
        for _ in range(number_of_repetitions):
            downstream_accuracies += np.array(
                measure_model_clasification_performance_once(
                    model, real_x, real_y, test_x, test_y, n_samples=n_samples, **kwargs
                )
            )
        downstream_accuracies /= number_of_repetitions
        results.loc[-1] = [dataset_name] + downstream_accuracies.tolist()
        results.index = results.index + 1
    return results


def measure_regresion_model_performance_once(
    model,
    real_x: pd.DataFrame,
    real_y: pd.DataFrame,
    test_x: pd.DataFrame,
    test_y: pd.DataFrame,
    n_samples: int | None = None,
    **kwargs,
) -> list[float]:
    synth_x, synth_y = model(
        real_x,
        real_y,
        n_samples=n_samples if n_samples else real_x.shape[0],
        **kwargs,
    )
    return [
        measure_random_forest_mean_absolute_error([synth_x], [synth_y], test_x, test_y)[
            0
        ],
        measure_xgb_mean_absolute_error([synth_x], [synth_y], test_x, test_y)[0],
        measure_tab_pfn_mean_absolute_error([synth_x], [synth_y], test_x, test_y)[0],
        measure_linear_regresion_mean_absolute_error(
            [synth_x], [synth_y], test_x, test_y
        )[0],
    ]


def measure_regresion_model_performance(
    model, number_of_repetitions: int = 5, n_samples: int | None = None, **kwargs
):
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
        downstream_accuracies = np.array([0.0, 0.0, 0.0, 0.0])
        train, test = dataset_getter()
        train = train[: min(10000, len(train))]  # CUDA out of memory error prevention
        real_x, real_y = (
            train.drop(REGRESION_TARGET, axis=1),
            train[REGRESION_TARGET].to_numpy(),
        )
        test_x, test_y = (
            test.drop(REGRESION_TARGET, axis=1),
            test[REGRESION_TARGET].to_numpy(),
        )
        for _ in range(number_of_repetitions):
            downstream_accuracies += np.array(
                measure_regresion_model_performance_once(
                    model, real_x, real_y, test_x, test_y, n_samples=n_samples, **kwargs
                )
            )
        downstream_accuracies /= number_of_repetitions
        results.loc[-1] = [dataset_name] + downstream_accuracies.tolist()
        results.index = results.index + 1
    return results


def measure_k_anonimity_once(
    model,
    real_x: pd.DataFrame,
    real_y: pd.DataFrame,
    n_samples: int | None = None,
    **kwargs,
) -> float:
    synth_x, _ = model(
        real_x,
        real_y,
        n_samples=n_samples if n_samples else real_x.shape[0],
        **kwargs,
    )
    return calculate_k_anonimity_for_datset(
        pd.DataFrame(synth_x, columns=real_x.columns)
    )


def measure_k_anonimity(
    model, number_of_repetitions: int = 5, n_samples: int | None = None, **kwargs
):
    results = pd.DataFrame(columns=[DATASET_COLUMN, "k_anonimity"])
    for dataset_name, dataset_getter in AVAILABLE_DATASETS.items():
        train, _ = dataset_getter()
        real_x, real_y = (
            train.drop(CLASYFICATION_TARGET, axis=1),
            train[CLASYFICATION_TARGET].to_numpy(),
        )
        single_dataset_k_values = []
        for _ in range(number_of_repetitions):
            single_dataset_k_values.append(
                measure_k_anonimity_once(
                    model, real_x[:1000], real_y[:1000], n_samples=n_samples, **kwargs
                )
            )
        results.loc[-1] = [dataset_name, np.mean(single_dataset_k_values)]
        results.index = results.index + 1
    return results


def measure_distance_to_nearest_neighbour_once(
    model,
    real_x: pd.DataFrame,
    real_y: pd.DataFrame,
    n_samples: int | None = None,
    **kwargs,
) -> dict[str, float]:
    synth_x, _ = model(
        real_x,
        real_y,
        n_samples=n_samples if n_samples else real_x.shape[0],
        **kwargs,
    )
    return calculate_distance_to_nearest_neighbour(
        pd.DataFrame(synth_x, columns=real_x.columns)
    )


def measure_distance_to_nearest_neighbour(
    model, number_of_repetitions: int = 5, n_samples: int | None = None, **kwargs
) -> pd.DataFrame:
    results = pd.DataFrame(columns=[DATASET_COLUMN, "mean", "std", "median"])
    for dataset_name, dataset_getter in AVAILABLE_DATASETS.items():
        train, _ = dataset_getter()
        real_x, real_y = (
            train.drop(CLASYFICATION_TARGET, axis=1),
            train[CLASYFICATION_TARGET].to_numpy(),
        )
        single_dataset_distances = pd.DataFrame(columns=["mean", "std", "median"])
        for _ in range(number_of_repetitions):
            single_dataset_distances.loc[-1] = (
                measure_distance_to_nearest_neighbour_once(
                    model, real_x[:1000], real_y[:1000], n_samples=n_samples, **kwargs
                ).values()
            )
            single_dataset_distances.index = single_dataset_distances.index + 1
        results.loc[-1] = [dataset_name, *(single_dataset_distances.mean().to_list())]
        results.index = results.index + 1
    return results


def measure_model_synthetic_data_coverage(
    model, number_of_repetitions: int = 5, n_samples: int | None = None, **kwargs
):
    results = pd.DataFrame(
        columns=[DATASET_COLUMN, "class_coverage_mean", "class_coverage_std"]
    )

    for dataset_name, dataset_getter in AVAILABLE_DATASETS.items():
        train, _ = dataset_getter()
        real_x, real_y = (
            train.drop(CLASYFICATION_TARGET, axis=1),
            train[CLASYFICATION_TARGET].to_numpy(),
        )
        single_dataset_coverages = []
        for _ in range(number_of_repetitions):
            synth_x, synth_y = model(
                real_x,
                real_y,
                n_samples=n_samples if n_samples else real_x.shape[0],
                **kwargs,
            )
            synth_x_df = pd.DataFrame(synth_x, columns=real_x.columns)
            synth_y_series = pd.Series(synth_y, name=CLASYFICATION_TARGET)
            synth_data = pd.concat([synth_x_df, synth_y_series], axis=1)
            real_data = pd.concat(
                [real_x, pd.Series(real_y, name=CLASYFICATION_TARGET)], axis=1
            )
            coverage_per_class = calculate_synthetic_data_coverage(
                real_data,
                synth_data,
                classification_target=CLASYFICATION_TARGET,
            )
            mean_coverage = np.mean(list(coverage_per_class.values()))
            single_dataset_coverages.append(mean_coverage)

        results.loc[-1] = [
            dataset_name,
            np.mean(single_dataset_coverages),
            np.std(single_dataset_coverages),
        ]
        results.index = results.index + 1
    return results


def calculate_synthetic_data_statistics(
    model, number_of_repetitions: int = 5, n_samples: int | None = None, **kwargs
) -> dict[str, pd.DataFrame]:
    results = []
    for dataset_name, dataset_getter in AVAILABLE_DATASETS.items():
        train, _ = dataset_getter()
        real_x, real_y = (
            train.drop(CLASYFICATION_TARGET, axis=1),
            train[CLASYFICATION_TARGET].to_numpy(),
        )
        single_dataset_statistics = []
        single_dataset_covariance = []
        single_dataset_ranks = []
        for _ in range(number_of_repetitions):
            synth_x, synth_y = model(
                real_x,
                real_y,
                n_samples=n_samples if n_samples else real_x.shape[0],
                **kwargs,
            )
            synth_x_df = pd.DataFrame(synth_x, columns=real_x.columns)
            synth_y_series = pd.Series(synth_y, name=CLASYFICATION_TARGET)
            synth_data = pd.concat([synth_x_df, synth_y_series], axis=1)
            real_data = pd.concat(
                [real_x, pd.Series(real_y, name=CLASYFICATION_TARGET)], axis=1
            )

            comparison_dsc = measure_dataset_statistics(real_data, synth_data)
            single_dataset_statistics.append(comparison_dsc)

            cov, rank = measure_matrix_statistics(real_x, synth_x_df)
            single_dataset_covariance.append(cov)
            single_dataset_ranks.append(rank)
        avg_comparison_dsc = (
            pd.concat(single_dataset_statistics).groupby(level=0).mean()
        )
        avg_covariance = pd.concat(single_dataset_covariance).groupby(level=0).mean()
        avg_rank = np.mean(single_dataset_ranks)
        results[dataset_name] = {
            "stats": avg_comparison_dsc,
            "cov": avg_covariance,
            "rank": avg_rank,
        }
    return results
