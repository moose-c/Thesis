
from models.EnKF_state_param import EnKF_state_param
from data.importInput import times, measurements, constants, location
from tools import EnKF_result_dictionary, MSE, run_model


import numpy as np
from matplotlib import pyplot as plt
from copy import deepcopy

# geen effect: c1, c2
# effect: cd
# c_cp_exp grote onzekerheid, logisch wand waarde cp = 8333, in verg tot cd = 250
start = 0
stop = 365
amt = 1

mock_data = run_model(start=start, stop=stop, variance=constants["Sigma_y"])
measurements = mock_data

init_constants = constants

for param_value in [0.5]:
    for vary in [["c_cd_exp", "c_cp_exp"]]:
        constants = deepcopy(init_constants)
        for i, param in enumerate(vary):
            constants[param] = np.log(param_value)

        uncalibrated = run_model(start=start, stop=stop, constants=constants)

        param_lsts = {}
        for param in vary:
            param_lsts[param] = []

        MSEs_cal = []
        for i in range(amt):
            (
                ensemble_predictions,
                ensemble_optimums,
                Sigma_xy_lst,
                Sigma_yy_lst,
                Kalman_lst,
            ) = EnKF_state_param(
                n=200,
                start=start,
                stop=stop,
                vary=vary,
                constants=constants,
                measurements=measurements,
            )
            results = EnKF_result_dictionary(
                ensemble_predictions, ensemble_optimums, vary
            )
            for param in vary:
                param_lsts[param].append((results["constants"][param]))

            constants = results["constants"]
            calibrated = run_model(start=start, stop=stop, constants=constants)
            MSEs_cal.append(MSE(calibrated, measurements[start:stop]))
        for param in vary:
            print(
                f"{param} from {param_value} becomes equal to: {(sum(np.exp(param_lsts[param])) / amt)}"
            )

        MSE_uncal = MSE(uncalibrated, measurements[start:stop])
        MSE_cal = sum(MSEs_cal) / amt

        print(
            f"the MSE with of {vary, param_value} uncalibrated constants: {MSE_uncal}"
        )
        print(f"the MSE with of {vary, param_value} calibrated constants: {MSE_cal}")

        print(f"{vary, param_value} now has ratio: {MSE_uncal / MSE_cal}")

    plt.plot(times[start:stop], measurements[start:stop], label="data")
    plt.plot(times[start:stop], uncalibrated, label="uncalibrated", alpha=0.5)
    plt.plot(times[start:stop], calibrated, label="calibrated")
    title = plt.title(f"Location {location}, Artificial Data Parameter Estimation")
    plt.figtext(0, 0, f"cd = {constants['cd']}, cp = {constants['cp']})")
    plt.xlabel("time (days)")
    plt.legend()
    plt.savefig(f"images/{title._text}.png")
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for param in vary:
        plt.plot(
            times[start:stop],
            np.exp(results[param + "_means"]),
            label=param.replace("_exp", ""),
        )
        ax.fill_between(
            times[start:stop],
            np.exp(results[param + "_means"] - np.sqrt(results[param + "_variances"])),
            np.exp(results[param + "_means"] + np.sqrt(results[param + "_variances"])),
            color="b",
            alpha=0.1,
        )
    plt.plot(times, [1] * len(times), alpha=0.4)
    title = plt.title(f"Location {location}, Artificial Data, Evolution of parameters")
    plt.figtext(
        0,
        0,
        f"cd = {np.exp(constants['c_cd_exp'])}, cp = {np.exp(constants['c_cp_exp'])}",
    )
    plt.xlabel("time (days)")
    plt.yscale("log")
    plt.legend()
    plt.savefig(f"images/{title._text}.png")
    plt.show()

    Kalman_state = [K[0] for K in Kalman_lst]
    for i, param in enumerate(vary):
        results[param + "_Kalman"] = [K[i + 1] for K in Kalman_lst]

    title = plt.title("Evolution of c_s_exp and c_c_exp elements of Kalman Gain matrix")
    for param in vary:
        plt.plot(
            times[start : stop - 1],
            results[param + "_Kalman"],
            label="Kalman Gain entry for " + param,
        )
    plt.legend()
    plt.savefig(f"images/{title._text}.png")
    plt.show()