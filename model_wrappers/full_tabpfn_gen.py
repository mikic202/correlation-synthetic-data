from tabpfn_extensions import unsupervised
from tabpfn import TabPFNClassifier, TabPFNRegressor
import numpy as np
import pandas as pd
import torch


class FullTabpfnGen(unsupervised.TabPFNUnsupervisedModel):
    def __init__(self, device):
        super().__init__(
            tabpfn_clf=TabPFNClassifier(device=device),
            tabpfn_reg=TabPFNRegressor(device=device),
        )

    def __call__(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        n_samples: int,
        attribute_names: list[str] | None = None,
        indices: list[int] | None = None,
        temp=1.0,
        **kwargs
    ):
        X_train["trarget"] = y_train
        data = torch.tensor(X_train.to_numpy())

        if indices is None or attribute_names is None:
            categorical_features = X_train.columns.to_list()

        else:
            feature_names = [attribute_names[i] for i in indices]
            categorical_features = [
                feature_names.index(name)
                for name in attribute_names
                if name in feature_names
            ]
        self.set_categorical_features(categorical_features)
        self.fit(data)

        synthetic_data = self.generate_synthetic_data(
            n_samples=n_samples,
            t=temp,
        )
        return synthetic_data
