from tabpfn_extensions import unsupervised
from tabpfn import TabPFNClassifier, TabPFNRegressor
import numpy as np
import pandas as pd


class FullTabpfnGen(unsupervised.UnsupervisedTabPFN):
    def __init__(self, *args, **kwargs):
        super().__init__(
            args=args,
            **kwargs,
            tabpfn_clf=TabPFNClassifier(),
            tabpfn_reg=TabPFNRegressor()
        )

    def __call__(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        n_samples: int,
        attribute_names: list[str] | None = None,
        indices: list[int] | None = None,
        temp=1.0,
    ):
        feature_names = [attribute_names[i] for i in indices]

        data = pd.concat([X_train, y_train], axis=1).reindex(X_train.index)

        if indices is not None or attribute_names is not None:
            categorical_features = data.columns.to_list()

        else:
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
