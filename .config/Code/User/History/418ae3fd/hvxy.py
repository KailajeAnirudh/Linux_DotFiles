import numpy as np

obseration_buffer = []
process_noise = np.diag([l1, l2]) 
measurement_noise = np.diag([a, b])
for i, observation in enumerate(obseration_buffer):
    
    mu_k1k = mu_k1
    sigma_k1k = F @ sigma_k1 @ F.T + np.random.multivariate_normal(np.zeros(2), process_noise)
    innovation = observation.vel - A @ (observation.state - mu_k1k)
    K = 