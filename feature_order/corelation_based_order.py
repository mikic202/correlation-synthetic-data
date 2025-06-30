import numpy as np
import pandas as pd


def generate_basic_order_of_features(
    feature_correlation: pd.DataFrame, correlation_treshold: int = 0.2
) -> pd.DataFrame:
    feature_correlation = abs(feature_correlation)
    mask = feature_correlation < correlation_treshold
    feature_correlation[mask] = 0.0
    np.fill_diagonal(feature_correlation.values, 0.0)
    return (
        pd.DataFrame(
            {
                "count": (feature_correlation > 0).sum(),
                "max": feature_correlation.max(),
                "min": feature_correlation.mask(feature_correlation <= 0).min(),
                "sum": feature_correlation.sum(),
            }
        )
        .fillna(0)
        .sort_values(by=["count", "max", "min", "sum"])
    )
