from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.metrics import roc_auc_score


def measure_logistic_regresion_auc(
    synthetic_x: list[pd.DataFrame],
    synthetic_y: list[list[int]],
    reral_x: pd.DataFrame,
    real_y: list[int],
):

    areas_under_curve = []
    for synt_x, synth_y in zip(synthetic_x, synthetic_y):
        logistic_regresion_clasifier = LogisticRegression().fit(synt_x, synth_y)
        areas_under_curve.append(
            roc_auc_score(
                real_y, logistic_regresion_clasifier.predict_proba(reral_x)[:, 1]
            )
        )
    return areas_under_curve
