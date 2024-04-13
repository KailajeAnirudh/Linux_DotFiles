# %% Imports

import numpy as np
from numpy.linalg import norm
from scipy.spatial.transform import Rotation
from scipy.constants import g


# %%



def complementary_filter_update(initial_rotation, angular_velocity, linear_acceleration, dt):
    """
    Implements a complementary filter update

    :param initial_rotation: rotation_estimate at start of update
    :param angular_velocity: angular velocity vector at start of interval in radians per second
    :param linear_acceleration: linear acceleration vector at end of interval in meters per second squared
    :param dt: duration of interval in seconds
    :return: final_rotation - rotation estimate after update
    """
    
    w_dt = angular_velocity*dt
    R_1k = initial_rotation*Rotation.from_rotvec(w_dt)

    g_prime = R_1k.apply(linear_acceleration)
    g_prime /= np.linalg.norm(g_prime) #Normalizing ||g'|| = 1 due to drift
    
    
    w_acc = np.cross(g_prime, np.array([1,0,0])) #Getting the axis along with the rotation must occur
    w_acc /= np.linalg.norm(w_acc) #Normalizing the vector to get a unit vector
    theta = np.arccos(np.dot(g_prime, np.array([1,0,0]))) #Get the angle through which the rotation must happen.
    quat = np.append(np.sin(theta/2)*w_acc, np.cos(theta/2))
    delta_q_acc =Rotation.from_quat(quat) #Rotation required assuming the |linear_acceleration/g| = 1

    get_alpha = lambda error_factor: min(1, max(0, -10*(error_factor-0.1)+1))
    alpha = get_alpha(abs(norm(linear_acceleration)/g-1))

    delta_q_prime_acc = (1-alpha)*np.array([0,0,0,1]) + alpha*delta_q_acc.as_quat()
    delta_q_prime_acc /= np.linalg.norm(delta_q_prime_acc)

    R_corrected = Rotation.from_quat(delta_q_prime_acc)*R_1k


    return R_corrected

