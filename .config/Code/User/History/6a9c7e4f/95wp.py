# # %% Imports

# import numpy as np
# from numpy.linalg import norm
# from scipy.spatial.transform import Rotation
# from scipy.constants import g


# # %%



# def complementary_filter_update(initial_rotation, angular_velocity, linear_acceleration, dt):
#     """
#     Implements a complementary filter update

#     :param initial_rotation: rotation_estimate at start of update
#     :param angular_velocity: angular velocity vector at start of interval in radians per second
#     :param linear_acceleration: linear acceleration vector at end of interval in meters per second squared
#     :param dt: duration of interval in seconds
#     :return: final_rotation - rotation estimate after update
#     """

#     # TODO Your code here - replace the return value with one you compute
    
#     w_dt = angular_velocity*dt
#     R_1k = Rotation.from_rotvec(w_dt)*initial_rotation

#     g_prime = R_1k.apply(linear_acceleration)
#     g_prime /= np.linalg.norm(g_prime) #Normalizing ||g'|| = 1 due ot drift
#     # g_prime[:] = g_prime[1], g_prime[2], g_prime[0] #Rotating the IMU measurement along the it's y axis, so that the IMU x-axis aligns with the gravity vector
    
#     w_acc = np.cross(g_prime, np.array([1,0,0]))
#     w_acc /= np.linalg.norm(w_acc)
#     theta = np.arccos(np.dot(g_prime, np.array([1,0,0])))
#     quat = np.append(np.sin(theta/2)*w_acc, np.cos(theta/2))
#     delta_q_acc =Rotation.from_quat(quat)

#     q_I = Rotation.from_quat([0,0,0,1]) #Null quaternion

#     get_alpha = lambda error_factor: min(1, max(0, -10*(error_factor-0.1)+1))
#     alpha = get_alpha(abs(norm(linear_acceleration)/g-1))

#     delta_q_prime_acc = (1-alpha)*q_I.as_quat() + alpha*delta_q_acc.as_quat()
#     delta_q_prime_acc /= np.linalg.norm(delta_q_prime_acc)

#     R_corrected = Rotation.from_quat(delta_q_prime_acc)*R_1k


#     return R_corrected


# %% Imports

import numpy as np
from numpy.linalg import norm
from scipy.spatial.transform import Rotation
import scipy


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

    # TODO Your code here - replace the return value with one you compute
    # print("ir",initial_rotation.as_quat())
    # print("av",angular_velocity)
    # print("la",linear_acceleration)
    # print("dt",dt)

    ex                    = np.array([1,0,0])
    w                     = angular_velocity
    w_hat                 = np.array([[0,-w[2],w[1]],
                                      [w[2],0,-w[0]],
                                      [-w[1],w[0],0]])
    R_prev_new            = Rotation.from_matrix(scipy.linalg.expm(w_hat*dt))
    R_prev_new_quat       = R_prev_new.as_quat()
    R_prev_new_quat       = R_prev_new_quat/norm(R_prev_new_quat)
    initial_rotation_quat = initial_rotation.as_quat()
    initial_rotation_quat = initial_rotation_quat/norm(initial_rotation_quat)
    R_one_new_quat        = quat_prod(initial_rotation_quat,R_prev_new_quat)
    g_dash                = quat_prod(quat_prod(R_one_new_quat,np.append(linear_acceleration,0)),quat_conj(R_one_new_quat))
    g_dash                = g_dash/norm(g_dash)
    w_acc                 = np.cross(g_dash[0:3],ex)
    w_acc                 = w_acc/norm(w_acc)
    omega                 = np.arccos(np.dot(g_dash[0:3],ex))
    q_acc                 = np.append(np.sin(omega/2)*w_acc, np.cos(omega/2))
    g                     = 9.9
    # print(norm(linear_acceleration))
    em                    = abs(norm(linear_acceleration)/g-1)
    if em<=0.1:
        alpha = 1
    elif em>0.1 and em<=0.2:
        alpha = -10*em + 2
    else:
        alpha = 0

    if q_acc[3]>0.5:
        q_acc_final = LERP(q_acc,alpha)
    else:
        q_acc_final = SLERP(q_acc,alpha)

    q_final = quat_prod(q_acc_final,R_one_new_quat)
    
    # r = Rotation.from_quat(q_acc_final)
    # R_acc = r.as_matrix()
    # R_final = R_acc @ Rotation.from_quat(R_one_new_quat).as_matrix()

    # R_prev_new = # matrix exponential
    # Convert to quaternion
    # R_one_new = # initial_rotation*R_prev_new     # Quaternion multiplication
    # g_dash = R_one_new*linear_acceleration        # Using Quaternions only
    # Normalize g_dash
    # w_acc = np.cross(g_dash,ex)
    # Normalize w_acc
    # omega = np.arccos(np.dot(g_dash,ex))
    # Construct required rotation = R_acc           # Using Quaternion only
        # Use SLERP and LERP
    # return R_acc*R_one_new

    R = Rotation.from_quat(q_final)
    # print(norm(q_final))
    return R

def quat_prod(u,v):
    """
    Input:
        u - First Quaternion
        v - Second Quaternion
    Output
        q_quat - uv (quaternion multiplication) 
    """
    q_const = np.array([u[3]*v[3]-u[0:3]@v[0:3]])
    q_axis  = np.array([u[3]*v[0:3]+v[3]*u[0:3] + np.cross(u[0:3],v[0:3])])
    q_quat  = np.append(q_axis,q_const)

    return q_quat

def quat_conj(q):
    """
    Input
        q - quaternion
    Output
        q_star - conjugate of the input quaternion
    """
    q_star = np.append(-q[0:3],q[3])
    return q_star

def SLERP(q_acc, alpha):
    """
    Input
        q_acc - quaternion having high frequency noise
        alpha - factor by which noise needs to be reduced (1 - full noise , 0 - no noise)
    Output
        q_acc_final - quaternion with reduced effect of noise
    """
    q_i = np.array([0,0,0,1])
    omega = np.arccos(np.dot(q_i[0:3],q_acc[0:3]))
    q_acc_final = (np.sin((1-alpha)*omega)/np.sin(omega))*q_i + np.sin(alpha*omega)/np.sin(omega)*q_acc
    q_acc_final = q_acc_final/norm(q_acc_final)
    return q_acc_final



def LERP(q_acc, alpha):
    """
    Input
        q_acc - quaternion having high frequency noise
        alpha - factor by which noise needs to be reduced (1 - full noise , 0 - no noise)
    Output
        q_acc_final - quaternion with reduced effect of noise
    """
    q_i = np.array([0,0,0,1])
    q_acc_final = (1-alpha)*q_i + alpha*q_acc
    q_acc_final = q_acc_final/norm(q_acc_final)
    return q_acc_final