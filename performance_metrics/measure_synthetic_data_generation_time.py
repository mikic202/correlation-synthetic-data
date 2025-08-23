from typing import Callable
import pandas as pd
from time import time
from test_datasets.dataset_getters import CLASYFICATION_TARGET


def measure_synthetic_data_generation_time(
    model: Callable[[pd.DataFrame, pd.DataFrame, int], pd.DataFrame],
    dataset_getter: Callable[[None], tuple[pd.DataFrame, pd.DataFrame]],
    minimum_samples: int = 100,
    maximum_samples: int = 10000,
    measurement_resolution: int = 1000,
) -> pd.DataFrame:
    train, _ = dataset_getter()
    x_train, y_train = (
        train.drop(CLASYFICATION_TARGET, axis=1),
        train[CLASYFICATION_TARGET].to_list(),
    )
    generation_time = []
    for n_samples in range(
        minimum_samples, maximum_samples + 1, measurement_resolution
    ):
        start = time()
        model(x_train, y_train, n_samples)
        end = time()
        generation_time.append(end - start)

    return pd.DataFrame(
        {
            "n_samples": range(
                minimum_samples, maximum_samples + 1, measurement_resolution
            ),
            "generation_time": generation_time,
        }
    )
