# from typing import Optional, Tuple, List
import numpy as np
import pandas as pd

from fbprophet import Prophet
import logging
from sklearn.impute import KNNImputer
import os
import utils


class suppress_stdout_stderr(object):
    """
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    """

    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])


class ImputeByMean:
    def __init__(
        self,
        groups=[],
    ) -> None:
        self.groups = groups

    def fit_transform(self, signal: pd.Series) -> pd.Series:
        col_to_impute = signal.name
        index_signal = signal.index
        signal = signal.reset_index()
        imputed = utils.custom_groupby(signal, self.groups)[col_to_impute].apply(
            lambda x: x.fillna(x.mean())
        )

        imputed = imputed.to_frame()
        imputed = imputed.fillna(0)
        imputed = imputed.set_index(index_signal)
        imputed = imputed[col_to_impute]
        return imputed

    def get_hyperparams(self):
        return {}


class ImputeByMedian:
    def __init__(
        self,
        groups=[],
    ) -> None:
        self.groups = groups

    def fit_transform(self, signal: pd.Series) -> pd.Series:
        col_to_impute = signal.name
        index_signal = signal.index
        signal = signal.reset_index()
        imputed = utils.custom_groupby(signal, self.groups)[col_to_impute].apply(
            lambda x: x.fillna(x.median())
        )

        imputed = imputed.to_frame()
        imputed = imputed.fillna(0)
        imputed = imputed.set_index(index_signal)
        imputed = imputed[col_to_impute]
        return imputed

    def get_hyperparams(self):
        return {}


class ImputeByMode:
    def __init__(
        self,
    ) -> None:
        pass

    def fit(self, signal: pd.Series) -> None:
        self.signal = signal
        self.imputed = self.signal.fillna(self.signal[self.signal.notnull()].mode()[0])

    def get_hyperparams(self):
        return {}


class RandomImpute:
    def __init__(
        self,
    ) -> None:
        pass

    def fit_transform(self, signal: pd.Series) -> pd.Series:
        imputed = signal.copy()
        number_missing = imputed.isnull().sum()
        obs = imputed[imputed.notnull()]
        imputed.loc[imputed.isnull()] = np.random.choice(
            obs.values, number_missing, replace=True
        )
        return imputed

    def get_hyperparams(self):
        return {}


class ImputeLOCF:
    def __init__(
        self,
        groups=[],
    ) -> None:
        self.groups = groups

    def fit_transform(self, signal: pd.Series) -> pd.Series:
        col_to_impute = signal.name
        index_signal = signal.index
        signal = signal.reset_index()
        imputed = utils.custom_groupby(signal, self.groups)[col_to_impute].apply(
            lambda x: x.ffill()
        )

        imputed = imputed.to_frame()
        imputed = imputed.fillna(0)
        imputed = imputed.set_index(index_signal)
        imputed = imputed[col_to_impute]
        # if index of missing values is 0, impute by the TS median
        return imputed.fillna(np.nanmedian(imputed))

    def get_hyperparams(self):
        return {}


class ImputeNOCB:
    def __init__(
        self,
        groups=[],
    ) -> None:
        self.groups = groups

    def fit_transform(self, signal: pd.Series) -> pd.Series:
        col_to_impute = signal.name
        index_signal = signal.index
        signal = signal.reset_index()
        imputed = utils.custom_groupby(signal, self.groups)[col_to_impute].apply(
            lambda x: x.bfill()
        )

        imputed = imputed.to_frame()
        imputed = imputed.fillna(0)
        imputed = imputed.set_index(index_signal)
        imputed = imputed[col_to_impute]
        # if index of missing values is 0, impute by the TS median
        return imputed.fillna(np.nanmedian(imputed))

    def get_hyperparams(self):
        return {}


class ImputeKNN:
    def __init__(self, **kwargs) -> None:
        for name, value in kwargs.items():
            setattr(self, name, value)

    def fit_transform(self, signal: pd.Series) -> pd.Series:
        # self.signal = signal.reset_index()
        # missing_indices = self.signal[self.signal.isnull()].index
        # for i in missing_indices:
        #     imputer = KNNImputer(n_neighbors=self.k)
        #     self.imputed = imputer.fit_transform( self.signal.loc[:i+1])
        # res = pd.Series([a[0] for a in self.imputed])

        self.signal = np.asarray(signal).reshape(-1, 1)
        imputer = KNNImputer(n_neighbors=self.k)
        res = imputer.fit_transform(self.signal)
        res = pd.Series([a[0] for a in res], index=signal.index)
        return res.fillna(np.nanmedian(res))

    def get_hyperparams(self):
        return {"k": self.k}


# does not work with kedro...
class ImputeProphet:
    def __init__(self, **kwargs) -> None:
        for name, value in kwargs.items():
            setattr(self, name, value)

    def fit_transform(self, signal: pd.Series) -> pd.Series:
        col_to_impute = signal.name
        data = pd.DataFrame()
        data["ds"] = signal.index.get_level_values("datetime")
        data["y"] = signal.values

        prophet = Prophet(
            daily_seasonality=self.daily_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            yearly_seasonality=self.yearly_seasonality,
            interval_width=self.interval_width,
        )
        with suppress_stdout_stderr():
            prophet.fit(data)

        forecast = prophet.predict(data[["ds"]])["yhat"]
        imputed = data["y"].fillna(forecast)
        imputed = imputed.to_frame(col_to_impute)
        imputed = imputed.set_index(signal.index)
        imputed = imputed[col_to_impute]
        return imputed

    def get_hyperparams(self):
        return {
            "daily_seasonality": self.daily_seasonality,
            "weekly_seasonality": self.weekly_seasonality,
            "yearly_seasonality": self.yearly_seasonality,
            "interval_width": self.interval_width,
        }
