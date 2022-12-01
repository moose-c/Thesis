from models.groundwater_model import groundwater_exponential
from data.importInput import prec, h_d, h_p, measurements, constants
from copy import deepcopy

import numpy as np
from numpy.random import normal


def EnKF_state_param(
    n=100, t=182, vary=["c_s"], constants=constants, measurements=measurements
):
    # initialize
    ensemble_init, ensemble_constants = initialization(n, vary, constants=constants)

    ensemble_pred = deepcopy(ensemble_init)
    ensemble_opt = deepcopy(ensemble_init)
    ensemble_predictions, ensemble_optimums = (
        [ensemble_init.copy()],
        [ensemble_init.copy()],
    )
    for day, measurement in enumerate(measurements[1 : t + 1]):
        ensemble_pred = deepcopy(ensemble_opt)
        ensemble_pred, ensemble_opt, ensemble_constants = EnKF_sp_step(
            day, measurement, ensemble_pred, ensemble_opt, ensemble_constants, n, vary
        )

        ensemble_predictions.append(deepcopy(ensemble_pred))
        ensemble_optimums.append(deepcopy(ensemble_opt))

    return ensemble_predictions, ensemble_optimums


def initialization(n, vary, constants=constants):
    Sigma_m = constants["Sigma_m"]
    Sigma_y = constants["Sigma_y"]

    constantvalue_lst = []
    for param in vary:
        constantvalue_lst.append(constants[param])

    ensemble_init = []
    for i in range(n):
        ensemble_init.append(
            np.array(
                [
                    min(
                        normal(measurements[0], Sigma_m),
                        constants["sl"],
                    ),
                ]
                + [
                    max(param_value + normal(0, param_value / 2.5), 0.01)
                    for param_value in constantvalue_lst
                ]
            )
        )

    ensemble_constants = [constants.copy() for i in range(n)]
    for i in range(n):
        for j, param in enumerate(vary):
            ensemble_constants[i][param] = ensemble_init[i][j + 1]
    return ensemble_init, ensemble_constants


def EnKF_sp_step(
    day, measurement, ensemble_pred, ensemble_opt, ensemble_constants, n, vary
):
    Sigma_m = constants["Sigma_m"]
    Sigma_y = constants["Sigma_y"]
    # Predictions for all previous optima
    for i, ensemble_member_opt in enumerate(ensemble_opt):
        ensemble_pred[i][0], _, _, _ = groundwater_exponential(
            ensemble_member_opt[0], prec[day], h_d[day], h_p[day], ensemble_constants[i]
        )
        ensemble_pred[i][0] += normal(0, Sigma_m)
        ensemble_pred[i][0] = min(ensemble_pred[i][0], constants["sl"])

    # Usually translation of model to measured variables
    h_pred = np.array([ensemble_pred[i][0] for i in range(n)])

    # Define Covariance Matrices
    pred_mean = h_pred.mean()
    Sigma_yy = 1 / (n - 1) * sum((h_pred - pred_mean) ** 2)

    res = 0
    ensemble_mean = np.array(1 / n * sum(np.array(ensemble_pred)))
    for i, prediction in enumerate(ensemble_pred):
        res += (prediction - ensemble_mean) * (h_pred[i] - pred_mean)
    Sigma_xy = 1 / (n - 1) * res

    # Add noise to Measurement
    meas_t = [measurement + normal(0, Sigma_y) for m in range(n)]

    # Fuse Prediction with Measurement
    alpha = 1
    K = Sigma_xy * ((Sigma_yy + Sigma_y) ** (-1))
    for j, pred in enumerate(h_pred):
        ensemble_opt[j] = ensemble_pred[j] + alpha * K * (meas_t[j] - pred)
    alpha = 0
    # Clip Ensemble Constants,
    for i, ensemble in enumerate(ensemble_opt):
        for j, param in enumerate(vary):
            ensemble_opt[i][j + 1] = np.clip(
                ensemble[j + 1],
                0.1 * constants[param],
                10 * constants[param],  # ensemble pread cap
            )
            ensemble_constants[i][param] = ensemble_opt[i][j + 1]

    return ensemble_pred, ensemble_opt, ensemble_constants