import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
import numpy as np
from heapq import nsmallest
from test_datasets.dataset_getters import CLASYFICATION_TARGET


MAX_NUMBER_OF_CLUSTERS = 20


def calculate_k_anonimity_for_datset(
    dataset: pd.DataFrame, identifier_atributes: list[str] | None = None
) -> float:
    if not identifier_atributes:
        identifier_atributes = dataset.columns.tolist()
    dataset = dataset[identifier_atributes]
    smallest_ks = []
    for n_clusters in range(2, MAX_NUMBER_OF_CLUSTERS + 1):
        clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = clusterer.fit_predict(dataset)
        silhouette_avg = silhouette_score(dataset, cluster_labels)
        smallest_ks.append(
            (silhouette_avg, np.unique(cluster_labels, return_counts=True)[1].min())
        )
    return nsmallest(
        1,
        nsmallest(MAX_NUMBER_OF_CLUSTERS // 5, smallest_ks, key=lambda x: -x[0]),
        key=lambda x: x[1],
    )[0][1]


def calculate_distance_to_nearest_neighbour(
    dataset: pd.DataFrame,
    identifier_atributes: list[str] | None = None,
) -> dict[str, float]:
    if not identifier_atributes:
        identifier_atributes = dataset.columns.tolist()
    dataset = dataset[identifier_atributes]
    model = NearestNeighbors(n_neighbors=2)
    model.fit(dataset)
    distances, _ = model.kneighbors(dataset)
    return {
        "mean": np.mean(distances[:, 1]),
        "std": np.std(distances[:, 1]),
        "median": np.median(distances[:, 1]),
    }


def measure_privacy(model, dataset_getter, n_samples: int | None = None):
    train, _ = dataset_getter()
    x_train, y_train = (
        train.drop(CLASYFICATION_TARGET, axis=1),
        train[CLASYFICATION_TARGET].to_list(),
    )
    synth_x, synth_y = model(
        x_train,
        y_train,
        n_samples=n_samples if n_samples else x_train.shape[0],
        balance_classes=True,
    )
    synth_data = pd.DataFrame(synth_x, columns=x_train.columns)
    synth_data[CLASYFICATION_TARGET] = pd.Series(synth_y)
    print(calculate_distance_to_nearest_neighbour(synth_data))
    print(calculate_k_anonimity_for_datset(synth_data))
