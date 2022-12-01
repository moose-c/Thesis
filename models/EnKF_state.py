from models.groundwater_model import groundwater_exponential
from data.importInput import prec, h_d, h_p, measurements, constants

import numpy as np
from numpy.random import normal


def EnKF_state(n=200):
    Sigma_m = constants["Sigma_m"]
    Sigma_y = constants["Sigma_y"]

    h_init = [normal(measurements[0], Sigma_y) for i in range(n)]
    h_init = np.array([min(h, constants["sl"]) for h in h_init])
    h_pred = h_init.copy()
    h_opt = h_init.copy()
    noisy_measurements, h_predictions, h_optimums = (
        [h_init.copy()],
        [h_pred.copy()],
        [h_opt.copy()],
    )
    for j, measurement in enumerate(measurements[1:]):
        # Predictions for all previous optima
        for i, optima in enumerate(h_opt):
            h_pred[i], _, _, _ = groundwater_exponential(
                optima, prec[j], h_d[j], h_p[j], constants
            )
            h_pred[i] += normal(0, Sigma_m)
            h_pred[i] = min(h_pred[i], constants["sl"])
        # Usually translation of model to measured variables

        # Define Covariance Matrices
        pred_mean = h_pred.mean()
        Sigma_x = 1 / (n - 1) * sum((h_pred - pred_mean) ** 2)
        Sigma_xy = Sigma_x

        # Add noise to Measurement
        meas_t = [measurement + normal(0, Sigma_y) for m in range(n)]

        # Fuse Prediction with Measurement
        K = Sigma_xy * ((Sigma_y + Sigma_x) ** (-1))
        for j, pred in enumerate(h_pred):
            h_opt[j] = pred + K * (meas_t[j] - pred)

        noisy_measurements.append(meas_t)
        h_predictions.append(h_pred.copy()), h_optimums.append(h_opt.copy())
    return h_predictions, h_optimums, noisy_measurements