from models.groundwater_model import groundwater_exponential
from data.importInput import times, prec, h_d, h_p, measurements, constants

import numpy as np
from numpy.random import normal


def KMFilter1D():
    h_opt = measurements[0]
    Sigma_m = constants["Sigma_m"]
    Sigma_y = constants["Sigma_y"]
    S_opt = Sigma_y
    h_predictions, h_optimums = [h_opt], [h_opt]
    S_predictions, S_optimums = [S_opt], [S_opt]
    for i, measurement in enumerate(measurements[1:]):
        # prediction
        h_pred, F, _, _ = groundwater_exponential(
            h_opt, prec[i], h_d[i], h_p[i], constants
        )
        S_pred = F**2 * S_opt + Sigma_m
        # measurement fusion
        meas_pred = measurement
        K_t = S_pred / (S_pred + Sigma_y)
        h_opt = h_pred + K_t * (meas_pred - h_pred)
        S_opt = (1 - K_t) * S_pred

        h_predictions.append(h_pred), h_optimums.append(h_opt)
        S_predictions.append(S_pred), S_optimums.append(S_opt)
    return (
        np.array(h_predictions),
        np.array(h_optimums),
        np.array(S_predictions),
        np.array(S_optimums),
    )