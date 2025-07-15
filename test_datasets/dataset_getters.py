import pandas as pd
import openml
from sklearn.model_selection import train_test_split


def get_pc4_dataset(test_size: float = 0.1) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset, _, _, _ = openml.datasets.get_dataset(1049).get_data(
        dataset_format="dataframe"
    )
    return train_test_split(dataset, test_size=test_size)


def get_mfeat_zernike_dataset(
    test_size: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset, _, _, _ = openml.datasets.get_dataset(22).get_data(
        dataset_format="dataframe"
    )
    dataset = dataset.rename(columns={"class": "c"}).astype("float32")
    return train_test_split(dataset, test_size=test_size)


def get_climate_model_simulation_dataset(
    test_size: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset, _, _, _ = openml.datasets.get_dataset(40994).get_data(
        dataset_format="dataframe"
    )
    dataset = dataset.rename(columns={"outcome": "c"}).astype("float32")
    return train_test_split(dataset, test_size=test_size)


def get_wdbc_dataset(test_size: float = 0.1) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset, _, _, _ = openml.datasets.get_dataset(1510).get_data(
        dataset_format="dataframe"
    )
    dataset = dataset.rename(columns={"Class": "c"}).astype("float32")
    return train_test_split(dataset, test_size=test_size)


def get_analcatdata_authorship_dataset(
    test_size: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset, _, _, _ = openml.datasets.get_dataset(458).get_data(
        dataset_format="dataframe"
    )
    dataset["Author"] = dataset["Author"].factorize()[0]
    dataset = dataset.rename(columns={"Author": "c"}).astype("float32")
    return train_test_split(dataset, test_size=test_size)
