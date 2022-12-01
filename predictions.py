import matplotlib.pyplot as plt
from copy import deepcopy
from math import sqrt
import numpy as np
import matplotlib.dates as mdates


from models.EnKF_state_param import EnKF_state_param
from tools import run_model, EnKF_result_dictionary, MSE
from data.importInput import constants, times, measurements, location


# set inportant values
start_calibration = 0
stop_calibration = 242
start_prediction = stop_calibration + 1
stop_prediction = 365
if constants["cp"] > constants["cd"]:
    vary = ["c_s_exp", "c_cd_exp"]
else:
    vary = ["c_s_exp", "c_cp_exp"]

# obtain calibrated parameters
ensemble_predictions, ensemble_optimums, _, _, _ = EnKF_state_param(
    vary=vary, start=start_calibration, stop=stop_calibration, n=200
)
results = EnKF_result_dictionary(ensemble_predictions, ensemble_optimums, vary)
constants = deepcopy(results["constants"])

# get the 2 predictions
uncalibrated = run_model(start=start_prediction, stop=stop_prediction)
calibrated = run_model(
    start=start_prediction, stop=stop_prediction, constants=constants
)

# compare results

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(times, measurements, label="measurements")
plt.plot(
    times[start_prediction:],
    uncalibrated,
    label="model with uncalibrated parameters",
    alpha=0.5,
)
plt.plot(times[start_prediction:], calibrated, label="model with calibrated parameters")
for i in [0, 1]:
    for j in [0, 1]:
        constants[vary[0]] = results["constants"][vary[0]] + (-1) ** i * np.sqrt(
            results[vary[0] + "_variances"][-1]
        )
        constants[vary[1]] = results["constants"][vary[1]] + (-1) ** j * np.sqrt(
            results[vary[1] + "_variances"][-1]
        )
        newoutput = run_model(
            start=start_prediction, stop=stop_prediction, constants=constants
        )
        ax.fill_between(
            times[start_prediction:],
            calibrated,
            newoutput,
            color="b",
            alpha=0.1,
        )
title = plt.title(
    f"Location {location}, calibrated until day {stop_calibration}, then start prediction"
)
plt.xlabel("time (days)")
plt.legend()
plt.savefig(f"images/{title._text}.png")
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
for param in vary:
    plt.plot(
        times[start_calibration:stop_calibration],
        np.exp(results[param + "_means"]) * 1,
        label=param.replace("_exp", ""),
    )
    ax.fill_between(
        times[start_calibration:stop_calibration],
        np.exp(results[param + "_means"] - np.sqrt(results[param + "_variances"])) * 1,
        np.exp(results[param + "_means"] + np.sqrt(results[param + "_variances"])) * 1,
        color="b",
        alpha=0.1,
    )
# ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
plt.yscale("log")
title = plt.title(
    f"Location {location}, Parameter evolution until day {stop_calibration}"
)
plt.xlabel("time (days)")
plt.legend()
plt.savefig(f"images/{title._text}.png")
plt.show()

RMSE_uncal = sqrt(MSE(uncalibrated, measurements[start_prediction:]))
RMSE_cal = sqrt(MSE(calibrated, measurements[start_prediction:]))

print(f"the RMSE with the calibrated constants: {RMSE_cal}")
print(f"the RMSE with the uncalibrated constants: {RMSE_uncal}")

print(RMSE_uncal / RMSE_cal)

constants = deepcopy(results["constants"])
c_s = np.exp(constants["c_s_exp"])
c_cp = np.exp(constants["c_cp_exp"])
c_cd = np.exp(constants["c_cd_exp"])
print(f"c_s = {c_s}, c_cp = {c_cp}, c_cd = {c_cd}")