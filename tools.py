from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import normal

from data.importInput import times, constants, measurements, prec, h_d, h_p
from models.groundwater_model import groundwater_exponential, groundwater_euler


def plot(measurements, h_opt, T=365):
    plt.plot(times[:T], measurements[:T], label="Measurements")
    plt.plot(times[:T], h_opt[:T], label="Optimal predictions")
    plt.legend()
    plt.show()


def MSE(list1, list2):
    return ((np.array(list1) - np.array(list2)) ** 2).mean()


def EnKF_result_dictionary(ensemble_predictions, ensemble_optimums, vary):
    """
    Function to compute the result from EnKF with state & parameter estimation

    returns:
        Dictionary filled with:
            - "h_predictions": list, for each day an ensemble of the predictions
            - "h_optima": list, for each day an ensemble of the optima
            - "param": list, for each param, for each day, an ensemble of the parameters
            - "h_pred_means": list, for each day the mean of the prediction ensemble
            - "h_opt_means": list, for each day the mean of the optima ensemble
            - "param_means": list, for each day the mean of the param ensemble
            - "param_variances": list, for each param, for each day, the sample variance of that parameter
            - "constants": dict, constants at final timestep.


    """
    results = {
        "h_predictions": [],
        "h_optima": [],
    } | {param: [] for param in vary}

    for t, day in enumerate(ensemble_optimums):
        for key in results.keys():
            results[key].append([])
        for i, ensemble in enumerate(day):
            results["h_predictions"][t].append(ensemble_predictions[t][i][0])
            results["h_optima"][t].append(ensemble[0])
            for i, param in enumerate(vary):
                results[param][t].append(ensemble[i + 1])
    results["h_opt_means"] = [np.array(day).mean() for day in results["h_optima"]]
    results["h_pred_means"] = [np.array(day).mean() for day in results["h_predictions"]]
    constantscopy = deepcopy(constants)
    for param in vary:
        results[param + "_means"] = np.array(
            [np.array(day).mean() for day in results[param]]
        )
        results[param + "_variances"] = np.array(
            [np.array(day).var() for day in results[param]]
        )
        constantscopy[param] = results[param + "_means"][-1]
    results["constants"] = constantscopy
    return results


def param_uncertainty_fixed_time(results, t=182, p1="c_s_exp", p2="c_c"):
    p1_values = results[p1][t]
    p2_values = results[p2][t]
    plt.plot(p1_values, p2_values, ".")
    plt.xlim(left=0, right=15)
    plt.ylim(bottom=0, top=15)


def run_model(
    measurements=measurements,
    constants=constants,
    start=0,
    stop=365,
    model="exponential",
    variance=0,
    returnN=False,
    returns=False,
    error=False,
    points_day=1,
):
    if model == "exponential":
        model = groundwater_exponential
    elif model == "euler":
        model = groundwater_euler
    else:
        return "Incorrect model name"
    results = [measurements[start]]
    N_lst = []
    s_lst = []
    error_lst = []
    for i in range(start, stop - 1):
        for j in range(points_day):
            result, _, s, N = model(
                results[-1], prec[i], h_d[i], h_p[i], constants, dt=1 / points_day
            )
            results.append(result + normal(0, variance))
            N_lst.append(N)
            s_lst.append(s)
    N_lst.append(N_lst[-1])
    s_lst.append(s_lst[-1])
    if error:
        return results, error_lst
    if returnN & returns:
        return results, N_lst, s_lst
    elif returnN:
        return N_lst
    elif returns:
        return s_lst
    else:
        return results