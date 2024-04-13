import numpy as np

obseration_buffer = []

for i, observation in enumerate(obseration_buffer):
    dt = 0.005
    mu_k1k = mu_k1k
    sigma_k1k = F @ sigma_k1 @ F.T + np.
    innovation = observation.vel - A @ (observation.state - mu_k1k)
    K = 