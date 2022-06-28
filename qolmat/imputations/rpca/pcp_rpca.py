from __future__ import annotations
from multiprocessing.sharedctypes import Value
from typing import Optional
from xmlrpc.client import boolean

import numpy as np
from numpy.typing import ArrayLike, NDArray

from qolmat.imputations.rpca.rpca import RPCA
from qolmat.imputations.rpca import utils


class PcpRPCA(RPCA):
    """
    This class implements the basic RPCA decomposition using Alternating Lagrangian Multipliers.

    References
    ----------
    Candès, Emmanuel J., et al. "Robust principal component analysis."
    Journal of the ACM (JACM) 58.3 (2011): 1-37

    Parameters
    ----------
    mu: Optional
        parameter for the convergence and shrinkage operator
    lam: Optional
        penalizing parameter for the sparse matrix
    """

    def __init__(
        self,
        n_rows: Optional[int] = None,
        mu: Optional[float] = None,
        lam: Optional[float] = None,
        maxIter: Optional[int] = int(1e4),
        tol: Optional[float] = 1e-6,
        verbose: bool = False,
    ) -> None:

        super().__init__(n_rows=n_rows, maxIter=maxIter, tol=tol, verbose=verbose)
        self.mu = mu
        self.lam = lam

    def get_params(self):
        dict_params = super().get_params()
        dict_params["mu"] = self.mu
        dict_params["lam"] = self.lam
        return dict_params

    def get_params_scale(self, signal):
        D_init, _ = self._prepare_data(signal=signal)
        proj_D = utils.impute_nans(D_init, method="median")
        mu = np.prod(proj_D.shape) / (4.0 * utils.l1_norm(self.proj_D))
        lam = 1 / np.sqrt(np.max(self.proj_D.shape))
        dict_params = {"mu": mu, "lam": lam}
        return dict_params

    def fit_transform(
        self,
        signal: NDArray,
        return_basis: boolean = False
    ) -> PcpRPCA:
        """
        Compute the RPCA decomposition of a matrix based on the PCP method

        Parameters
        ----------
        signal : NDArray
            Observations
        """
        D_init, n_add_values = self._prepare_data(signal=signal)
        proj_D = utils.impute_nans(D_init, method="median")

        if self.mu is None:
            self.mu = np.prod(proj_D.shape) / (4.0 * utils.l1_norm(self.proj_D))

        if self.lam is None:
            self.lam = 1 / np.sqrt(np.max(self.proj_D.shape))

        D_norm = np.linalg.norm(proj_D, "fro")

        n, m = D_init.shape
        A = np.zeros((n, m))
        Y = np.zeros((n, m))

        errors = []
        for iteration in range(self.maxIter):
            X = utils.svd_thresholding(proj_D - A + Y / self.mu, 1 / self.mu)
            A = utils.soft_thresholding(proj_D - X + Y / self.mu, self.lam / self.mu)
            Y += self.mu * (proj_D - X - A)

            errors.append(np.linalg.norm(proj_D - X - A, "fro") / D_norm)
            if errors[-1] <= self.tol:
                if self.verbose:
                    print(f"Converged in {iteration} iterations")
                break
        if return_basis:
            U, _, Vh = np.linalg.svd(X, full_matrices=False, compute_uv=True)
            result = [U, Vh]
        else:
            result = []

        if n_add_values > 0:
            X.flat[-n_add_values:] = np.nan
            A.flat[-n_add_values:] = np.nan

        if not((len(signal.shape) == 1) or (signal.shape[1] in [0, 1])):
            result = [X, A, errors] + result
        else:
            result = [X.flatten(), A.flatten(), errors] + result
        return tuple(result)
