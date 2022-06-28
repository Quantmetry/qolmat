# from typing import Optional, Tuple, List
import numpy as np
import pandas as pd

# from fbprophet import Prophet
import logging

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
from qolmat.imputations.rpca.pcp_rpca import RPCA
from qolmat.imputations.rpca.temporal_rpca import OnlineTemporalRPCA, TemporalRPCA
from qolmat.benchmark import utils
import os
import sys


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


class ImputeColumnWise:
    def __init__(
        self,
        groups=[],
    ) -> None:
        self.groups = groups

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:

        col_to_impute = df.columns
        imputed = df.copy()
        for col in col_to_impute:
            # df_col = df[col].reset_index()
            imputed[col] = self.fit_transform_col(df[col]).values
        imputed.fillna(0, inplace=True)
        return imputed

    def get_hyperparams(self):
        return {}


class ImputeByMean(ImputeColumnWise):
    def __init__(
        self,
        groups=[],
    ) -> None:
        super().__init__(groups=groups)

    def fit_transform_col(self, signal: pd.Series) -> pd.Series:
        col = signal.name
        signal = signal.reset_index()
        # imputed = utils.custom_groupby(signal, self.groups)[[col_to_impute]].apply(lambda x: x.fillna(x.mean()))
        imputed = signal[col].fillna(
            utils.custom_groupby(signal, self.groups)[col].transform("mean")
        )
        return imputed


class ImputeByMedian(ImputeColumnWise):
    def __init__(
        self,
        groups=[],
    ) -> None:
        super().__init__(groups=groups)

    def fit_transform_col(self, signal: pd.Series) -> pd.Series:
        col = signal.name
        signal = signal.reset_index()
        # imputed = utils.custom_groupby(signal, self.groups)[[col_to_impute]].apply(lambda x: x.fillna(x.mean()))
        imputed = signal[col].fillna(
            utils.custom_groupby(signal, self.groups)[col].transform("median")
        )
        return imputed


class RandomImpute(ImputeColumnWise):
    def __init__(
        self,
    ) -> None:
        pass

    def fit_transform_col(self, signal: pd.Series) -> pd.Series:
        col = signal.name
        imputed = signal.reset_index()
        number_missing = imputed[col].isnull().sum()
        obs = imputed.loc[imputed[col].notnull(), col].values
        imputed.loc[imputed[col].isnull(), col] = np.random.choice(
            obs, number_missing, replace=True
        )
        return imputed[col]


class ImputeLOCF(ImputeColumnWise):
    def __init__(
        self,
        groups=[],
    ) -> None:
        super().__init__(groups=groups)

    def fit_transform_col(self, signal: pd.Series) -> pd.Series:
        col = signal.name
        imputed = signal.reset_index()
        imputed = utils.custom_groupby(imputed, self.groups)[col].transform(
            lambda x: x.ffill()
        )
        return imputed.fillna(np.nanmedian(imputed))


class ImputeNOCB(ImputeColumnWise):
    def __init__(
        self,
        groups=[],
    ) -> None:
        super().__init__(groups=groups)

    def fit_transform_col(self, signal: pd.Series) -> pd.Series:
        col = signal.name
        imputed = signal.reset_index()
        imputed = utils.custom_groupby(imputed, self.groups)[col].transform(
            lambda x: x.bfill()
        )
        return imputed.fillna(np.nanmedian(imputed))


class ImputeKNN(ImputeColumnWise):
    def __init__(self, **kwargs) -> None:
        for name, value in kwargs.items():
            setattr(self, name, value)

    def fit_transform_col(self, signal: pd.Series) -> pd.Series:
        col = signal.name
        signal = signal.reset_index()
        imputed = np.asarray(signal[col]).reshape(-1, 1)
        imputer = KNNImputer(n_neighbors=self.k)
        imputed = imputer.fit_transform(imputed)
        imputed = pd.Series([a[0] for a in imputed], index=signal.index)
        return imputed.fillna(np.nanmedian(imputed))

    def get_hyperparams(self):
        return {"k": self.k}


class ImputeByInterpolation(ImputeColumnWise):
    def __init__(self, **kwargs) -> None:
        for name, value in kwargs.items():
            setattr(self, name, value)

    def fit_transform_col(self, signal: pd.Series) -> pd.Series:
        col = signal.name
        signal = signal.reset_index()
        print(signal.datetime.isna().sum())
        signal = signal.set_index("datetime")
        return signal[col].interpolate(method=self.method)


class ImputeBySpline(ImputeColumnWise):
    def __init__(
        self,
    ) -> None:
        pass

    def fit_transform_col(self, signal: pd.Series) -> pd.Series:
        return signal.interpolate(option="spline")


"""
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
"""


class ImputeRPCA:
    def __init__(
        self, method, multivariate=False, **kwargs
    ) -> None:
        self.multivariate = multivariate

        if method == "PCP":
            self.rpca = RPCA()
        elif method == "temporal":
            self.rpca = TemporalRPCA()
        elif method == "online":
            self.rpca = OnlineTemporalRPCA()
            
        for name, value in kwargs.items():
            setattr(self.rpca, name, value)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:

        if self.multivariate:
            imputed, _, _ = self.rpca.fit_transform(signal=df.values)
            imputed = pd.DataFrame(imputed, columns=df.columns, index = df.index)
        else:
            imputed = pd.DataFrame()
            for col in df.columns:
                imputed_signal, _, _ = self.rpca.fit_transform(
                    signal=df[col].values
                )
                imputed[col] = imputed_signal
        imputed.index = df.index
        
        return imputed

    def get_hyperparams(self):
        pass


class ImputeIterative:
    def __init__(self, **kwargs) -> None:
        # self.initial_strategy = "median"
        # self.imputation_order = "ascending"
        # for name, value in kwargs.items():
        #     setattr(self, name, value)
        self.kwargs = kwargs

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        ii = IterativeImputer(
            **(self.kwargs)
            # initial_strategy=self.initial_strategy,
            # imputation_order=self.imputation_order,
        )
        res = ii.fit_transform(df.values)
        imputed = pd.DataFrame(columns=df.columns)
        for ind, col in enumerate(imputed.columns):
            imputed[col] = res[:, ind]
        imputed.index = df.index
        return imputed

    def get_hyperparams(self):
        pass
