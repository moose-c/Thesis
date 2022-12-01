import sys

sys.path.append(r"C:\Users\moos.castelijn\Documents\Scriptie")

import numpy as np
from numpy.random import normal

from data.importInput import BOFEK


def groundwater_euler(h_opt, p, h_d, h_p, constants, dt=1):
    """
    Iterative model for the groundwater


            Parameters:
                    level (float): Current groundwater level
                    q (float): Groundwater entering the cell from bellow (recharge)
                    p (float): Groundwater entering the cell from above (precipitation)

            Returns:
                    predict (float): prediction of the groundwater according to the model
    Euler solution is not correct.. Now it is!! Model validated check.

    """
    groundtype = constants["groundtype"]
    sl = constants["sl"]
    cd = constants["cd"]
    c1 = constants["c1"]
    c2 = constants["c2"]

    s = storage(h_opt, groundtype, sl)

    F = 1 - (dt / s) * (1 / c1 + 1 / c2 + 1 / cd)
    B = dt / s * np.array([1 / cd, 1, (c1 + c2) / (c1 * c2)])
    u = np.array([[h_d], [p], [h_p]])

    h_pred = min((F * h_opt + float(np.matmul(B, u))), sl)
    return h_pred, F, B, u


def groundwater_exponential(h_opt, p, h_d, h_p, constants, dt=1):
    # Retrieving Constants
    groundtype = constants["groundtype"]
    sl = constants["sl"]
    c_c = constants["c_c"]
    cd = constants["cd"] * c_c
    c1 = constants["c1"] * c_c
    c2 = constants["c2"] * c_c
    c_s = constants["c_s"]

    # Calculate Storage for this h
    s = c_s * storage(h_opt, groundtype, sl)

    # defining matrices and vectors for Model Calculations
    c = 1 / cd + 1 / c1 + 1 / c2
    d = np.exp(-dt * c / s)
    F = d
    B = 1 / c * (1 - d) * np.array([1 / cd, 1, (c1 + c2) / (c1 * c2)])
    u = np.array([[h_d], [p], [h_p]]) + [[normal(0, 0)], [0], [0]]

    # Model Calculations, removed: + normal(0, Sigma_m)
    h_pred = min((F * h_opt + float(np.matmul(B, u))), sl)
    return h_pred, F, B, u


def storage(h, groundtype, sl):
    if h >= sl:
        f = -5
    else:
        a, b, c, d = BOFEK[groundtype]
        num1 = a * b
        num2 = c * ((sl - h) * 100) ** d
        den = 1 / (b + ((sl - h) * 100) ** d)
        f = (num1 + num2) * den
    return np.clip(f, 0.01, 0.9)