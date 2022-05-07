from audioop import rms
from lib2to3.pytree import Base
from math import floor
from typing import Optional, Union

import numpy as np
import pandas as pd
from numpy.random import RandomState
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.utils import check_random_state, resample

class EvaluateImputor:

    def __init__(self,
                signal: ArrayLike = None,
                D: ArrayLike = None,
                prop: float = 0.1,
                cv: int = 3,
                random_state: Optional[Union[int, RandomState]] = None
                ):
                
                if (D is None) + (signal is None) != 1:
                    raise ValueError("Should have only D or Signal")
                
                self.signal = signal
                self.D = D
                if self.signal is not None:
                    indices_to_nan = signal.loc[signal > 0].index
                elif self.D is not None:
                    flatten_D = D.flatten()
                    indices_to_nan = flatten_D.loc[~np.isnan(flatten_D)].index
                
                random_state = check_random_state(random_state)
                
                nan_subsets = []
                
                for _ in range(cv):
                    nan_subsets.append(
                            resample(
                                indices_to_nan,
                                replace=False,
                                n_samples=floor(len(indices_to_nan) * prop),
                                random_state=random_state,
                                stratify=None,
                        )
                    )
                self.nan_subsets = nan_subsets
                self.random_state = random_state

    def eval(y_true, y_pred):
        rmse = mean_squared_error(y_true, y_pred, squared = False)
        mae = mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred)/np.abs(y_true)))
        wmape = np.mean(np.abs(y_true - y_pred))/np.mean(np.abs(y_true))
        return rmse, mae, mape, wmape

    def scores_imputor(self, imputor: BaseEstimator, func = eval):

        if self.signal is not None:
            input = self.signal.copy()
            transform = self.signal.copy()
        elif self.D is not None:
            input = self.D.copy()
            transform = self.D.copy()
        RMSE=[]
        MAE=[]
        MAPE=[]
        WMAPE=[]

        for nan_indices in self.nan_subsets:
            if len(transform.shape) == 1:
                transform.iloc[nan_indices] = np.nan
                impute, _, _ = imputor.fit_transform(signal = transform.values)
                impute = pd.Series(impute, index = transform.index)
            else:
                transform.flat().iloc[nan_indices] = np.nan
                impute, _, _ = imputor.fit_transform(D =transform.values)
            rmse, mae, mape, wmape = func(input.loc[nan_indices], impute.loc[nan_indices])
            RMSE.append(rmse)
            MAE.append(mae)
            MAPE.append(mape)
            WMAPE.append(wmape)

            return {"rmse" : np.mean(RMSE),
                    "mae" : np.mean(MAE),
                    "mape" : np.mean(MAPE),
                    "wmape" : np.mean(WMAPE)}