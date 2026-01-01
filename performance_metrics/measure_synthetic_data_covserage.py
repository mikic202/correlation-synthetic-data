import pandas as pd
from scipy.spatial import Delaunay, ConvexHull
from sklearn.decomposition import PCA
import numpy as np


def calculate_synthetic_data_coverage(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    classification_target: str = "c",
):
    class_labels = real_data[classification_target].unique()
    hulls_per_class = {}
    pca_transforms_per_class = {}
    for label in class_labels:
        pca = PCA(n_components=min(7, real_data.shape[1] - 1))
        transformed_points = pca.fit_transform(
            real_data[real_data[classification_target] == label].drop(
                columns=[classification_target]
            )
        )
        pca_transforms_per_class[label] = pca
        hulls_per_class[label] = ConvexHull(transformed_points)

    coverage_per_class = {}
    for label in class_labels:
        pca = pca_transforms_per_class[label]
        hull = hulls_per_class[label]
        synthetic_points = synthetic_data[
            synthetic_data[classification_target] == label
        ].drop(columns=[classification_target])
        transformed_synthetic_points = pca.transform(synthetic_points)
        inside_count = 0
        delaunay = Delaunay(hull.points[hull.vertices])
        inside_count = np.sum(delaunay.find_simplex(transformed_synthetic_points) >= 0)
        coverage_per_class[label] = (
            inside_count / len(synthetic_points) if len(synthetic_points) > 0 else 0.0
        )
    return coverage_per_class
