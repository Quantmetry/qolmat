from __future__ import annotations
from typing import Optional, Tuple, List, Type

import numpy as np
import skopt

from numpy.typing import NDArray

from qolmat.rpca import RPCA
from qolmat.utils import utils



class GraphRPCA(RPCA):
    """
    This class implements Fast Robust PCA on Graphs using the FISTA algorithm

    References
    ----------
    Shahid, Nauman, et al. "Fast robust PCA on graphs."
    IEEE Journal of Selected Topics in Signal Processing 10.4 (2016): 740-756.

    Parameters
    ----------
    n_rows : Optional[int]
        period/seasonality of the signal
    rank : Optional[int]
        (estimated) low-rank of the matrix D
    gamma1 : int
        regularizing parameter for the graph G1, constructed from the columns of D
    gamma2 : int
        regularizing parameter for the graph G1, constructed from the rows of D
    G1 : Optional[np.ndarray]
        graph G1, constructed from the columns of D
    G2 : Optional[np.ndarray]
        graph G2, constructed from the rows of D
    nbg1 : Optional[int]
        number of closest neighbors to construct graph G1, default=10
    nbg2 : Optional[int]
        number of closest neighbors to construct graph G2, default=10
    maxIter: int, default = 1e4
        maximum number of iterations taken for the solvers to converge
    tol: float, default = 1e-6
        tolerance for stopping criteria
    verbose: bool, default = False
    """

    def __init__(
        self,
        n_rows: Optional[int] = None,
        rank: Optional[int] = None,
        gamma1: Optional[float] = None,
        gamma2: Optional[float] = None,
        G1: Optional[np.ndarray] = None,
        G2: Optional[np.ndarray] = None,
        nbg1: Optional[int] = 10,
        nbg2: Optional[int] = 10,
        maxIter: Optional[int] = int(1e4),
        tol: Optional[float] = 1e-6,
        verbose: Optional[bool] = False,
    ) -> None:

        super().__init__(n_rows=n_rows, maxIter=maxIter, tol=tol, verbose=verbose)
        self.rank = rank
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.G1 = G1
        self.G2 = G2
        self.nbg1 = nbg1
        self.nbg2 = nbg2

    def fit_transform(
        self,
        signal: NDArray
    ) -> None:
        """Compute the RPCA on graph.

        Parameters
        ----------
       signal : NDArray
            Observations
        """
        D_init, n_add_values = self._prepare_data(signal=signal)
        omega = ~np.isnan(D_init)
        proj_D = utils.impute_nans(D_init, method="median")

        if self.rank is None:
            self.rank = utils.approx_rank(proj_D)

        if self.G1 is None:
            self.G1 = utils.construct_graph((self.proj_D).T, n_neighbors=self.nbg1)
        if self.G2 is None:
            self.G2 = utils.construct_graph((self.proj_D), n_neighbors=self.nbg2)

        laplacian1 = utils.get_laplacian(self.G1)
        laplacian2 = utils.get_laplacian(self.G2)

        X = proj_D.copy()
        Y = proj_D.copy()
        t_past = 1

        lam = 1 / (
              2 * self.gamma1 * np.linalg.norm(laplacian1, 2)
            + 2 * self.gamma2 * np.linalg.norm(laplacian2, 2)
        )

        errors = np.full((self.maxIter,), np.nan, dtype = float)
        for iteration in range(self.maxIter):

            X_past = X.copy()
            Y_past = Y.copy()

            grad_g = 2 * (self.gamma1 * Y @ laplacian1 + self.gamma2 * laplacian2 @ Y)

            X = utils.proximal_operator(Y_past - lam * grad_g, proj_D, lam)
            t = (1 + (1 + 4 * t_past**2) ** 0.5) / 2
            Y = X + (t_past - 1) / t * (X - X_past)

            error = np.linalg.norm(Y - Y_past, "fro") / np.linalg.norm(Y_past, "fro")
            errors[iteration] = error

            if error < self.tol:
                if self.verbose:
                    print(
                        f"Converged in {iteration} iterations with an error equals to {error}."
                    )
                break

            t_past = t

        self.errors = errors

        A = D_init - X
        
        if not((len(signal.shape) == 1) or (signal.shape[1] in [0, 1])):
            result =  [X, A, errors]
        else:
            X = X.T
            A = A.T
            
            if n_add_values > 0:
                X.flat[-n_add_values:] = np.nan
                A.flat[-n_add_values:] = np.nan
            ts_X = X.flatten()
            ts_A = A.flatten()
            ts_X = ts_X[~np.isnan(ts_X)]
            ts_A = ts_A[~np.isnan(ts_A)]
            result = [ts_X, ts_A, errors]

        return tuple(result)


####################################################################
# TO DO
# construct graph with missing values
# important for the cross validation to work
####################################################################
