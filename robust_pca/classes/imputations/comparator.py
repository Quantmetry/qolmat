import pandas as pd
import numpy as np
import cross_validation
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
)  # , mean_absolute_percentage_error
from sklearn.utils import resample
from collections import defaultdict
from typing import Optional
from math import floor
from skopt.space import Categorical, Real, Integer
import matplotlib.pyplot as plt
import utils


class Comparator:
    def __init__(
        self,
        data,
        ratio_missing,
        models_to_test,
        cols_to_impute,
        search_params={},
        corruption="missing",
    ):

        self.df = data
        self.ratio_missing = ratio_missing
        self.cols_to_impute = cols_to_impute
        self.models_to_test = models_to_test
        self.search_params = search_params
        self.corruption = corruption

    def create_corruptions(self, signal: pd.Series, random_state: Optional[int] = 29):

        indices = np.where(signal.notna())[0]  # signal[signal.notna()].index

        self.indices = resample(
            indices,
            replace=False,
            n_samples=floor(len(indices) * self.ratio_missing),
            random_state=random_state,
            stratify=None,
        )

        self.corrupted_signal = signal.copy()
        if self.corruption == "missing":
            self.corrupted_signal[self.indices] = np.nan
        elif self.corruption == "outlier":
            self.corrupted_signal[self.indices] = np.random.randint(
                0, high=3 * np.max(signal), size=(int(len(signal) * self.ratio_missing))
            )

    def get_errors(
        self,
        signal_ref: pd.Series,
        signal_imputed: pd.Series,
    ) -> float:

        rmse = mean_squared_error(
            signal_ref[self.indices], signal_imputed[self.indices], squared=False
        )
        mae = mean_absolute_error(
            signal_ref[self.indices], signal_imputed[self.indices]
        )
        # mape = mean_absolute_percentage_error(signal_ref[self.indices], signal_imputed[self.indices])
        wmape = np.mean(
            np.abs(signal_ref[self.indices] - signal_imputed[self.indices])
        ) / np.mean(np.abs(signal_ref[self.indices]))
        return {"rmse": rmse, "mae": mae, "wmape": wmape}  # "mape": mape,

    def compare(self):

        results = {}
        for tested_model in self.models_to_test:
            search_space, search_name = utils.get_search_space(
                tested_model, self.search_params
            )
            res_intermediate = {}
            for col in self.cols_to_impute:
                series = self.df[col]
                errors = defaultdict(list)
                for _ in range(1):
                    random_state = np.random.randint(0, 10 * 9)
                    self.create_corruptions(series, random_state=random_state)
                    cv = cross_validation.CrossValidation(
                        tested_model,
                        search_space=search_space,
                        search_name=search_name,
                        ratio_missing=self.ratio_missing,
                        corruption=self.corruption,
                    )
                    imputed_signal = cv.fit_transform(self.corrupted_signal)
                    for k, v in self.get_errors(series, imputed_signal).items():
                        errors[k].append(v)
                res_intermediate[col] = {k: np.mean(v) for k, v in errors.items()}
            results[type(tested_model).__name__] = res_intermediate
        return results
