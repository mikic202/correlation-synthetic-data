import pandas as pd
from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from tabpfn import TabPFNClassifier
from sklearn.linear_model import LogisticRegression
import torch


NUMBER_OF_UNIQUE_ELEMENTS_FOR_BINARY_CLASIFICATION = 2


def measuer_roc(
    synthetic_x: list[pd.DataFrame],
    synthetic_y: list[list[int]],
    reral_x: pd.DataFrame,
    real_y: list[int],
    model_class,
    **kwargs,
):
    areas_under_curve = []
    for synt_x, synth_y in zip(synthetic_x, synthetic_y):
        clasifier = model_class(**kwargs).fit(synt_x, synth_y)
        if len(np.unique(real_y)) > NUMBER_OF_UNIQUE_ELEMENTS_FOR_BINARY_CLASIFICATION:
            areas_under_curve.append(
                roc_auc_score(
                    real_y,
                    clasifier.predict_proba(reral_x),
                    multi_class="ovr",
                )
            )
            continue
        areas_under_curve.append(
            roc_auc_score(real_y, clasifier.predict_proba(reral_x)[:, 1])
        )
    return areas_under_curve


def measure_logistic_regresion_auc(
    synthetic_x: list[pd.DataFrame],
    synthetic_y: list[list[int]],
    reral_x: pd.DataFrame,
    real_y: list[int],
):
    return measuer_roc(
        synthetic_x,
        synthetic_y,
        reral_x,
        real_y,
        LogisticRegression,
        solver="saga",
        penalty="l2",
        C=1.0,
        tol=1e-3,
        max_iter=500,
        n_jobs=-1,
        multi_class="auto",
    )


def measure_random_forest_auc(
    synthetic_x: list[pd.DataFrame],
    synthetic_y: list[list[int]],
    reral_x: pd.DataFrame,
    real_y: list[int],
):
    return measuer_roc(
        synthetic_x,
        synthetic_y,
        reral_x,
        real_y,
        RandomForestClassifier,
    )


def measure_tabpfn_auc(
    synthetic_x: list[pd.DataFrame],
    synthetic_y: list[list[int]],
    reral_x: pd.DataFrame,
    real_y: list[int],
):
    return measuer_roc(
        synthetic_x,
        synthetic_y,
        reral_x,
        real_y,
        TabPFNClassifier,
        n_estimators=len(reral_x.columns) * 2,
        device=("cuda" if torch.cuda.is_available() else "cpu"),
    )


def measure_xgb_auc(
    synthetic_x: list[pd.DataFrame],
    synthetic_y: list[list[int]],
    reral_x: pd.DataFrame,
    real_y: list[int],
):
    n_estimators = len(reral_x.columns) * 2
    max_depth = max(len(reral_x.columns) // 5, 4)

    return measuer_roc(
        synthetic_x,
        synthetic_y,
        reral_x,
        real_y,
        XGBClassifier,
        n_estimators=n_estimators,
        max_depth=max_depth,
    )
