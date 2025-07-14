import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score


def measure_xgb_auc(
    synthetic_x: list[pd.DataFrame],
    synthetic_y: list[list[int]],
    reral_x: pd.DataFrame,
    real_y: list[int],
):
    n_estimators = len(reral_x.columns) * 2
    max_depth = max(len(reral_x.columns) // 5, 4)

    areas_under_curve = []
    for synt_x, synth_y in zip(synthetic_x, synthetic_y):
        random_forest_clasifier = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=1,
            objective="binary:logistic",
        ).fit(synt_x, synth_y)
        areas_under_curve.append(
            roc_auc_score(real_y, random_forest_clasifier.predict_proba(reral_x)[:, 1])
        )
    return areas_under_curve
