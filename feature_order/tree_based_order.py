import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
import pandas as pd


def generate_tree_based_order_of_features(
    dataset: pd.DataFrame, ascending=True
) -> list[str]:
    feature_importances = np.zeros(len(dataset.columns))

    for i, feature in enumerate(dataset.columns):
        if dataset[feature].dtype == "int64":
            rf = RandomForestClassifier()
        else:
            rf = RandomForestRegressor()
        rf.fit(dataset.drop(feature, axis=1), dataset[feature])
        idx_to_update = list(range(i)) + list(range(i + 1, len(dataset.columns)))
        feature_importances[idx_to_update] += np.mean(
            [tree.feature_importances_ for tree in rf.estimators_], axis=0
        )
    return (
        list(reversed(dataset.columns[np.argsort(feature_importances)[::-1]].to_list()))
        if ascending
        else dataset.columns[np.argsort(feature_importances)].to_list()
    )


def generate_xgboost_based_order_of_features(
    dataset: pd.DataFrame, ascending=True
) -> list[str]:
    feature_importances = np.zeros(len(dataset.columns))

    n_estimators = len(dataset.columns) * 2
    max_depth = len(dataset.columns) // 2

    for i, feature in enumerate(dataset.columns):
        if dataset[feature].dtype == "int64":
            rf = XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=1,
                objective="binary:logistic",
            )
        else:
            rf = XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=1,
                objective="reg:squarederror",
            )
        rf.fit(dataset.drop(feature, axis=1), dataset[feature])
        idx_to_update = list(range(i)) + list(range(i + 1, len(dataset.columns)))
        feature_importances[idx_to_update] += rf.feature_importances_
    return (
        list(reversed(dataset.columns[np.argsort(feature_importances)[::-1]].to_list()))
        if ascending
        else dataset.columns[np.argsort(feature_importances)].to_list()
    )
