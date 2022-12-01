import csv
from datetime import datetime
import numpy as np

data = []
with open("data/timeseries/9012.csv") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        data.append(row)

data = data[1:]
# data has shape: [[time, prec, deep head, polder head, meas, BOFEK, surface level], [...], [...], ...]
times, prec, h_d, h_p, measurements, constants = (
    np.array([datetime.fromisoformat(row[0]) for row in data]),
    np.array([float(row[1]) for row in data]),
    np.array([float(row[2]) for row in data]),
    np.array([float(row[3]) for row in data]),
    np.array([float(row[4]) for row in data]),
    {
        "groundtype": float(data[0][5]),
        "sl": float(data[0][6]),
        "cd": float(data[0][7]),
        "c1": float(data[0][8]),
        "c2": float(data[0][9]),
        "Sigma_y": 0.0049,
        "Sigma_m": 0.0841,
        "c_s": 1,
        "c_c": 1,
    },
)

BOFEK = {
    # code: (a, b, c, d)
    101: (0.000303691, 1285.538567, 0.31755023, 1.293560239),
    102: (-0.00171752, 1568.493141, 1.028670946, 1.028078548),
    103: (-0.00144013, 1008.565908, 0.537106661, 1.12531779),
    104: (0.005371247, 40313.54772, 0.103202519, 2.549597467),
    105: (0.001939729, 1276.551716, 0.518408741, 1.172037814),
    106: (-0.002824644, 1063.402536, 0.444133433, 1.184539211),
    107: (0.00049918, 739.4457865, 0.373017707, 1.204034812),
    108: (-0.002829721, 800.5077888, 0.6564414, 1.03068723),
    109: (-0.002225621, 876.5315566, 1.180106953, 0.907188721),
    110: (0.001230167, 1197.579027, 0.608512682, 1.159873983),
}

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    plt.plot(times, h_d, label="deep head")
    plt.plot(times, h_p, label="polder head")
    plt.plot(times, [constants["sl"]] * len(times), label="surface level")
    plt.title("Input data")
    plt.figtext(
        0, 0, f"cd = {constants['cd']}, c1 = {constants['c1']}), c2 = {constants['c2']}"
    )
    plt.legend()
    plt.show()