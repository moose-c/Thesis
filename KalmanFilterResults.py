
from models.KalmanFilter import KMFilter1D
from data.importInput import times, measurements, constants
from tools import MSE, run_model

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Dit moet nog in verslag!
# 9024 was best mooi, 8890 ook, 8907 ook

h_predictions, h_optimums, S_predictions, S_optimums = KMFilter1D()
model_solution = run_model()

title = plt.title("Kalman Filter a priori state estimates, 1 year")
plt.plot(times, measurements, label="Measurements")
plt.plot(times, h_predictions, label="A priori state estimates")
plt.plot(times, model_solution, alpha=0.4, label="Model prediction")
plt.xlabel("time (days)")
plt.ylabel("depth (meter)")
plt.legend()
plt.savefig(f"images/{title._text}.png")
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(times[:31], measurements[:31], label="Measurements")
plt.plot(times[:31], h_predictions[:31], label="A priori state estimates")
plt.plot(times[:31], model_solution[:31], alpha=0.4, label="Model prediction")
title = plt.title("Kalman Filter a priori state estimates, 1 month")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
plt.xlabel("time (days)")
plt.ylabel("depth (meter)")
plt.legend()
plt.savefig(f"images/{title._text}.png")
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
title = plt.title("Kalman Filter a priori state error s. d. evolution, 1 month")
plt.plot(times[:31], measurements[:31], label="Measurements")
plt.plot(times[:31], h_predictions[:31], label="A priori state estimates")
ax.fill_between(
    times[:31],
    (h_predictions[:31] - np.sqrt(S_predictions[:31])),
    (h_predictions[:31] + np.sqrt(S_predictions[:31])),
    color="r",
    alpha=0.3,
)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
plt.xlabel("time (days)")
plt.ylabel("depth (meter)")
plt.ylim(-1.1, -0.2)
plt.legend()
plt.savefig(f"images/{title._text}.png")
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
title = plt.title("Kalman Filter a posteriori state estimates, 1 year")
plt.plot(times, measurements, label="Measurements")
plt.plot(times, h_optimums, label="A posteriori state estimates")
plt.plot(times, model_solution, alpha=0.4, label="Model prediction")
plt.xlabel("time (days)")
plt.ylabel("depth (meter)")
plt.legend()
plt.savefig(f"images/{title._text}.png")
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
title = plt.title("Kalman Filter a posteriori state estimates, 1 month")
plt.plot(times[:31], measurements[:31], label="Measurements")
plt.plot(times[:31], h_optimums[:31], label="A posteriori state estimates")
plt.plot(times[:31], model_solution[:31], alpha=0.4, label="Model prediction")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
plt.xlabel("time (days)")
plt.ylabel("depth (meter)")
plt.legend()
plt.savefig(f"images/{title._text}.png")
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
title = plt.title("Kalman Filter a posteriori state error s. d. evolution, 1 month")
plt.plot(times[:31], measurements[:31], label="Measurements")
plt.plot(times[:31], h_optimums[:31], label="A posteriori state estimates")
ax.fill_between(
    times[:31],
    (h_optimums[:31] - np.sqrt(S_optimums[:31])),
    (h_optimums[:31] + np.sqrt(S_optimums[:31])),
    color="r",
    alpha=0.3,
)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
plt.xlabel("time (days)")
plt.ylabel("depth (meter)")
plt.ylim(-1.1, -0.2)
plt.legend()
plt.savefig(f"images/{title._text}.png")
plt.show()

print("MSE predictions: " + str(MSE(measurements, h_predictions)))
print("MSE optimums: " + str(MSE(measurements, h_optimums)))

title = plt.title("Kalman Filter, Propagation of error standard deviation")
plt.plot(times, np.sqrt(S_predictions), label="A priori state error standard deviation")
plt.plot(
    times, np.sqrt(S_optimums), label="A posteriori state error standard deviation"
)
plt.xlabel("time (days)")
plt.ylabel("depth (meter)")
plt.legend()
plt.savefig(f"images/{title._text}.png")
plt.show()

Kalman_Gain = S_predictions / (S_predictions + constants["Sigma_y"])
title = plt.title("Kalman Filter, Kalman Gain matrix")
plt.plot(times, Kalman_Gain)
plt.legend()
plt.savefig(f"images/{title._text}.png")
plt.show()

print(Kalman_Gain.mean())
