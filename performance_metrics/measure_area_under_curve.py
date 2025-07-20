import pandas as pd
from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from tabpfn import TabPFNClassifier
from sklearn.linear_model import LogisticRegression


NUMBER_OF_UNIQUE_ELEMENTS_FOR_BINARY_CLASIFICATION = 2


def measure_logistic_regresion_auc(
    synthetic_x: list[pd.DataFrame],
    synthetic_y: list[list[int]],
    reral_x: pd.DataFrame,
    real_y: list[int],
):

    areas_under_curve = []
    for synt_x, synth_y in zip(synthetic_x, synthetic_y):
        logistic_regresion_clasifier = LogisticRegression().fit(synt_x, synth_y)
        if len(np.unique(synth_y)) > NUMBER_OF_UNIQUE_ELEMENTS_FOR_BINARY_CLASIFICATION:
            areas_under_curve.append(
                roc_auc_score(
                    real_y,
                    logistic_regresion_clasifier.predict_proba(reral_x),
                    multi_class="ovr",
                )
            )
            continue
        areas_under_curve.append(
            roc_auc_score(
                real_y, logistic_regresion_clasifier.predict_proba(reral_x)[:, 1]
            )
        )
    return areas_under_curve


def measure_random_forest_auc(
    synthetic_x: list[pd.DataFrame],
    synthetic_y: list[list[int]],
    reral_x: pd.DataFrame,
    real_y: list[int],
):
    areas_under_curve = []
    for synt_x, synth_y in zip(synthetic_x, synthetic_y):
        random_forest_clasifier = RandomForestClassifier().fit(synt_x, synth_y)
        if len(np.unique(synth_y)) > NUMBER_OF_UNIQUE_ELEMENTS_FOR_BINARY_CLASIFICATION:
            areas_under_curve.append(
                roc_auc_score(
                    real_y,
                    random_forest_clasifier.predict_proba(reral_x),
                    multi_class="ovr",
                )
            )
            continue
        areas_under_curve.append(
            roc_auc_score(real_y, random_forest_clasifier.predict_proba(reral_x)[:, 1])
        )
    return areas_under_curve


def measure_tabpfn_auc(
    synthetic_x: list[pd.DataFrame],
    synthetic_y: list[list[int]],
    reral_x: pd.DataFrame,
    real_y: list[int],
):
    areas_under_curve = []
    for synt_x, synth_y in zip(synthetic_x, synthetic_y):
        random_forest_clasifier = TabPFNClassifier().fit(synt_x, synth_y)
        if len(np.unique(synth_y)) > NUMBER_OF_UNIQUE_ELEMENTS_FOR_BINARY_CLASIFICATION:
            areas_under_curve.append(
                roc_auc_score(
                    real_y,
                    random_forest_clasifier.predict_proba(reral_x),
                    multi_class="ovr",
                )
            )
            continue
        areas_under_curve.append(
            roc_auc_score(real_y, random_forest_clasifier.predict_proba(reral_x)[:, 1])
        )
    return areas_under_curve


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
        if len(np.unique(synth_y)) > NUMBER_OF_UNIQUE_ELEMENTS_FOR_BINARY_CLASIFICATION:
            areas_under_curve.append(
                roc_auc_score(
                    real_y,
                    random_forest_clasifier.predict_proba(reral_x),
                    multi_class="ovr",
                )
            )
            continue
        areas_under_curve.append(
            roc_auc_score(real_y, random_forest_clasifier.predict_proba(reral_x)[:, 1])
        )
    return areas_under_curve
