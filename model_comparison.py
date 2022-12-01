from data.importInput import times, prec, measurements, h_d, h_p, constants
from models.groundwater_model import (
    groundwater_exponential,
    groundwater_euler,
)
from tools import MSE

import matplotlib.pyplot as plt


# use 1/2 year of data for calibration, 2nd half for testing

t = 365


h_euler_lst = [measurements[0]]
h_analytical_lst = [measurements[0]]

for i in range(t - 1):
    h_euler_lst.append(
        groundwater_euler(
            h_euler_lst[-1],
            prec[i],
            h_d[i],
            h_p[i],
            constants,
        )[0]
    )
    h_analytical_lst.append(
        groundwater_exponential(
            h_analytical_lst[-1],
            prec[i],
            h_d[i],
            h_p[i],
            constants,
        )[0]
    )

MSE_euler = MSE(measurements, h_euler_lst)
MSE_exponential = MSE(measurements, h_analytical_lst)
print(MSE_euler, MSE_exponential)

plt.plot(times[:t], h_euler_lst, label="Prediction from Euler model")
plt.plot(times[:t], measurements[:t], label=f"Measurements")
plt.title("Euler solution")
plt.legend()
plt.show()

plt.plot(times[:t], h_analytical_lst, label="Predicitons from analytical solution")
plt.plot(times[:t], measurements[:t], label=f"Measurements")
plt.title("Anaylical solution")
plt.legend()
plt.show()