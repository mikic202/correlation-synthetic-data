import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


def measure_random_forest_auc(
    synthetic_x: list[pd.DataFrame],
    synthetic_y: list[list[int]],
    reral_x: pd.DataFrame,
    real_y: list[int],
):
    areas_under_curve = []
    for synt_x, synth_y in zip(synthetic_x, synthetic_y):
        random_forest_clasifier = RandomForestClassifier().fit(synt_x, synth_y)
        areas_under_curve.append(
            roc_auc_score(real_y, random_forest_clasifier.predict_proba(reral_x)[:, 1])
        )
    return areas_under_curve
