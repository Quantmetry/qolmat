import gc
import datetime
from math import floor
import signal
from typing import Optional, Union

import skopt
from skopt import gp_minimize
from sklearn.metrics import make_scorer

import numpy as np
from numpy.random import RandomState
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.utils import check_random_state, resample

from .evaluate_imputer import EvaluateImputor
from .rpca import RPCA


class get_objective:

    def __init__(self,
                 imputor: RPCA,
                 imputor_eval: EvaluateImputor,
                 space,
                 scoring, #rmse
                 exp):
        self.imputor = imputor
        self.imputor_eval = imputor_eval
        self.space = space
        self.scoring = scoring
        self.exp = exp

    def fit_transform_and_evaluate(self):
        @skopt.utils.use_named_args(self.space)
        def obj_func(**all_params):
            
            if self.exp:
                all_params = dict(map(lambda x: (x[0], np.exp(x[1]*np.log(10))), all_params.items()))

            imputor = self.imputor.set_params(**all_params)
            scores = self.imputor_eval.scores_imputor(imputor)
            gc.collect()
            return -scores[self.scoring]

        return obj_func

def rpca_optimizer(imputor: RPCA,
                   imputor_eval: EvaluateImputor,
                   space,
                   scoring,  #rmse
                   exp_variables=False,
                   n_random_starts = 10, 
                   epoch = 100,
                   n_jobs = 1,
                   verbose = False,
                   return_signal = True):
                   
    objective = get_objective(imputor=imputor,
                imputor_eval=imputor_eval,
                space=space,
                scoring=scoring,
                exp = exp_variables
    )
    objective_func = objective.fit_transform_and_evaluate()

    print("beging_time = " + str(datetime.datetime.now().strftime("%H:%M:%S")))
    result = gp_minimize(
            func=objective_func,
            dimensions=space,
            acq_optimizer="lbfgs",
            random_state=imputor_eval.random_state,
            n_jobs=n_jobs,
            verbose=verbose,
            n_initial_points=n_random_starts,
            n_calls=epoch,
        )
    best_params = {space[index].name: result["x"][index] for index in range(len(result["x"]))}
    best_score = result["fun"]

    print("OPTIMIZATION FINISHED")
    print(f"best_params = {best_params}")
    print(f"best_scores = {best_score}")


    print("end_time = " + str(datetime.datetime.now().strftime("%H:%M:%S")))
    best_imputor = imputor.set_params(**best_params)
    gc.collect()
    if return_signal:
        if len(imputor_eval.signal.shape) == 1:
            transform_signal, _, _ = imputor.fit_transform(signal = imputor_eval.signal)
        else:
            transform_signal, _, _ = imputor.fit_transform(D = imputor_eval.D)
        return best_imputor, best_score, best_params, transform_signal
    return best_imputor, best_score, best_params

