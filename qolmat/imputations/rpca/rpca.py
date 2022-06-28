from __future__ import annotations
from typing import Optional
from xmlrpc.client import boolean

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, TransformerMixin

from qolmat.imputations.rpca.utils import utils


class RPCA(BaseEstimator, TransformerMixin):
    """
    This class is the root class of the RPCA methods.

    Parameters
    ----------
    n_rows: Optional
        Number of columns in case reshaping of the 1D signal.
    maxIter: int, default = 1e4
        maximum number of iterations taken for the solvers to converge
    tol: float, default = 1e-6
        tolerance for stopping criteria
    verbose: bool, default = False
    """

    def __init__(
        self,
        n_rows: Optional[int] = None,
        maxIter: Optional[int] = int(1e4),
        tol: Optional[float] = 1e-6,
        verbose: bool = False,
    ) -> None:

        self.n_rows = n_rows
        self.maxIter = maxIter
        self.tol = tol
        self.verbose = verbose

    def _prepare_data(self, signal: NDArray) -> None:
        """
        Prepare data for RPCA computation:
        Transform signal to matrix if needed
        Get the omega matrix
        Impute the nan values if needed
        """

        if (len(signal.shape) == 1) or (signal.shape[1] in [0, 1]):
            if len(signal.shape) == 2:
                signal = signal.flatten()

            self.n_rows = (
                utils.get_period(signal) if self.n_rows is None else self.n_rows
            )
            D_init, n_add_values = utils.signal_to_matrix(signal, n_rows=self.n_rows)
        else:
            D_init = signal.copy()
            n_add_values = 0
        return D_init, n_add_values

    def get_params(self):
        return {
            "n_rows": self.n_rows,
            "maxIter": self.maxIter,
            "tol": self.tol,
            "verbose": self.verbose,
        }

    def set_params(self, **kargs):
        for param_key in kargs.keys():
            if param_key in self.__dict__.keys():
                setattr(self, param_key, kargs[param_key])
        return self

    def fit_transform(
        self,
        signal: NDArray,
        return_basis: boolean = False
    ) -> RPCA:
        X, _ = self._prepare_data(signal=signal)
        A = np.zeros(X.shape, dtype=float)

        if not((len(signal.shape) == 1) or (signal.shape[1] in [0, 1])):
            result = [X, A]
        else:
            result = [X.flatten(), A.flatten()]
            
        if return_basis:
            U, _, Vh = np.linalg.svd(X, full_matrices=False, compute_uv=True)
            result += [U, Vh]
        return tuple(result)