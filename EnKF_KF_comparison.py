
from models.EnKF_state import EnKF_state
from models.KalmanFilter import KMFilter1D
from tools import MSE

import numpy as np

MSE_priori_state = []
MSE_posteriori_state = []
MSE_priori_sd = []
MSE_posteriori_sd = []
for i in range(50):
    (
        h_predictions_EnKF,
        h_optimums_EnKF,
        S_predictions_EnKF,
        S_optimums_EnKF,
    ) = EnKF_state()
    h_predictions_KF, h_optimums_KF, S_predictions_KF, S_optimums_KF = KMFilter1D()
    MSE_priori_state.append(MSE(h_predictions_KF, h_predictions_EnKF))
    MSE_posteriori_state.append(MSE(h_optimums_KF, h_optimums_EnKF))
    MSE_priori_sd.append(MSE(np.sqrt(S_predictions_KF), np.sqrt(S_predictions_EnKF)))
    MSE_posteriori_sd.append(MSE(np.sqrt(S_optimums_KF), np.sqrt(S_optimums_EnKF)))

print(np.array(MSE_priori_state).mean())
print(np.array(MSE_posteriori_state).mean())
print(np.array(MSE_priori_sd).mean())
print(np.array(MSE_posteriori_sd).mean())
