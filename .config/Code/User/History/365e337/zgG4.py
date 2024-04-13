# Imports

import numpy as np
from scipy.spatial.transform import Rotation


# %%

def estimate_pose(uvd1, uvd2, pose_iterations, ransac_iterations, ransac_threshold):
    """
    Estimate Pose by repeatedly calling ransac

    :param uvd1:
    :param uvd2:
    :param pose_iterations:
    :param ransac_iterations:
    :param ransac_threshold:
    :return: Rotation, R; Translation, T; inliers, array of n booleans
    """

    R = Rotation.identity()

    for i in range(0, pose_iterations):
        w, t, inliers = ransac_pose(uvd1, uvd2, R, ransac_iterations, ransac_threshold)
        R = Rotation.from_rotvec(w.ravel()) * R

    return R, t, inliers

def solve_w_t(uvd1, uvd2, R0):
    """
    solve_w_t core routine used to compute best fit w and t given a set of stereo correspondences

    :param uvd1: 3xn ndarray : normailzed stereo results from frame 1
    :param uvd2: 3xn ndarray : normailzed stereo results from frame 1
    :param R0: Rotation type - base rotation estimate
    :return: w, t : 3x1 ndarray estimate for rotation vector, 3x1 ndarray estimate for translation
    """

    # TODO Your code here replace the dummy return value with a value you compute
    w = t = np.zeros((3,1))
    A = np.empty((2*uvd1.shape[1], 6))
    b_stack = np.empty((2*uvd1.shape[1],1))
    for i, (uvd1_prime, uvd2_prime) in enumerate(zip(uvd1.T, uvd2.T)):
        lmatrix = np.array([[1, 0, -uvd1_prime[0]],
                            [0, 1, -uvd1_prime[1]]])
        y = R0.apply(np.hstack((uvd2_prime[:2], 1)))
        b = -lmatrix@y
        yhat_d2_matrix = np.array([[0,      y[2], -y[1], uvd2_prime[2],             0, 0],
                                   [-y[2],     0,  y[0],             0, uvd2_prime[2], 0],
                                   [y[1],  -y[0],     0,             0,             0, uvd2_prime[2]]])
        A[2*i:2*i+2, :] = lmatrix@yhat_d2_matrix
        b_stack[2*i:2*i+2, :] = b.reshape(2,1)

    guess_sol = np.linalg.lstsq(A, b_stack)[0]
    w = guess_sol[:3]
    t = guess_sol[3:]   

    return w, t


def find_inliers(w, t, uvd1, uvd2, R0, threshold):
    """

    find_inliers core routine used to detect which correspondences are inliers

    :param w: ndarray with 3 entries angular velocity vector in radians/sec
    :param t: ndarray with 3 entries, translation vector
    :param uvd1: 3xn ndarray : normailzed stereo results from frame 1
    :param uvd2:  3xn ndarray : normailzed stereo results from frame 2
    :param R0: Rotation type - base rotation estimate
    :param threshold: Threshold to use
    :return: ndarray with n boolean entries : Only True for correspondences that pass the test
    """
    n = uvd1.shape[1]
    inliers = np.empty(n, dtype=bool)
    w = w.flatten()
    t = t.flatten()
    w_hat = np.array([[0, -w[2], w[1]],
                  [w[2], 0, -w[0]],
                  [-w[1], w[0], 0]])
    
    for i, (uvd1_prime, uvd2_prime) in enumerate(zip(uvd1.T, uvd2.T)):
        delta = np.array([[1, 0, -uvd1_prime[0]],
                      [0,1,-uvd1_prime[1]]]) @ (np.eye(3)+w_hat)@(R0.apply(np.hstack((uvd2_prime[:2], 1)))+ uvd2_prime[2]*t)
        inliers[i] = np.linalg.norm(delta)<threshold

    return inliers


def ransac_pose(uvd1, uvd2, R0, ransac_iterations, ransac_threshold):
    """

    ransac_pose routine used to estimate pose from stereo correspondences

    :param uvd1: 3xn ndarray : normailzed stereo results from frame 1
    :param uvd2: 3xn ndarray : normailzed stereo results from frame 1
    :param R0: Rotation type - base rotation estimate
    :param ransac_iterations: Number of RANSAC iterations to perform
    :ransac_threshold: Threshold to apply to determine correspondence inliers
    :return: w, t : 3x1 ndarray estimate for rotation vector, 3x1 ndarray estimate for translation
    :return: ndarray with n boolean entries : Only True for correspondences that are inliers

    """
    n = uvd1.shape[1]


    # TODO Your code here replace the dummy return value with a value you compute
    w = t = np.zeros((3,1))
    max_inliers = 0
    
    for k in range(max(ransac_iterations, 1)):
        indeces = np.random.choice(np.arange(n), 6)
        uvd1_trial = uvd1[:,indeces]
        uvd2_trial = uvd2[:,indeces]
        w, t = solve_w_t(uvd1_trial, uvd2_trial, R0)
        inliers = find_inliers(w=w, t=t, 
                               uvd1=uvd1, uvd2=uvd2, R0=R0, 
                               threshold=ransac_threshold)
        if k == 0:
            best_guess = (w, k)
        if inliers.sum() > max_inliers:
            max_inliers = inliers.sum()
            best_guess = (w, t)

    w = best_guess[0]
    t = best_guess[1]
    return w, t, inliers
