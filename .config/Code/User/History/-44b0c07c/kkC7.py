import numpy as np
from scipy.spatial.transform import Rotation

class SE3Control(object):
    """

    """
    def __init__(self, quad_params):
        """
        This is the constructor for the SE3Control object. You may instead
        initialize any parameters, control gain values, or private state here.

        For grading purposes the controller is always initialized with one input
        argument: the quadrotor's physical parameters. If you add any additional
        input arguments for testing purposes, you must provide good default
        values!

        Parameters:
            quad_params, dict with keys specified by crazyflie_params.py

        """

        # Quadrotor physical parameters.
        self.mass            = quad_params['mass'] # kg
        self.Ixx             = quad_params['Ixx']  # kg*m^2
        self.Iyy             = quad_params['Iyy']  # kg*m^2
        self.Izz             = quad_params['Izz']  # kg*m^2
        self.arm_length      = quad_params['arm_length'] # meters
        self.rotor_speed_min = quad_params['rotor_speed_min'] # rad/s
        self.rotor_speed_max = quad_params['rotor_speed_max'] # rad/s
        self.k_thrust        = quad_params['k_thrust'] # N/(rad/s)**2
        self.k_drag          = quad_params['k_drag']   # Nm/(rad/s)**2

        # You may define any additional constants you like including control gains.
        self.inertia = np.diag(np.array([self.Ixx, self.Iyy, self.Izz])) # kg*m^2
        self.g = 9.81 # m/s^2

        # STUDENT CODE HERE
        self.Kp = np.diag([7.5, 7.5, 35.0])*1.05
        self.Kd = np.diag([4.5,4.5,12.0/1.025])/1.025
        
        self.Kr = np.diag([9200, 13200, 2200.0])
        self.Kw = np.diag([295, 445, 200.0])

      

        # self.Kp = np.diag([7, 7, 35.0])
        # self.Kd = np.diag([4.,4.,30.0])
        # self.Kr = np.diag([2900, 2900, 220.0])
        # self.Kw = np.diag([250, 250, 200.0])

        # self.Kp = np.diag([8, 8, 19.0])#/4.2
        # self.Kd = np.diag([5.5,5.5,9.0])#/2.5
        # self.Kr = np.diag([2800, 2800, 150.0])
        # self.Kw = np.diag([125, 125, 80.0])

        # self.Kp = np.diag([0.75, 0.75, 100.0])#/4.2
        # self.Kd = np.diag([2.5,2.5,40.0])*1.5#/2.5
        # self.Kr = np.diag([600, 600, 30.0])
        # self.Kw = np.diag([32.5, 32.5, 18.0])
        self.Kp_agg = np.diag([60, 60, 45.0])
        self.Kd_agg = np.diag([14, 14, 18.0])
        self.weight_vector = np.array([0, 0, self.mass * self.g])
        self.gamma = self.k_drag/self.k_thrust
        self.motor_speed_calc_matrix = np.linalg.inv(self.k_thrust*np.array([[1,1,1,1],
                                                 [0, self.arm_length, 0, -self.arm_length],
                                                 [-self.arm_length, 0 , self.arm_length, 0],
                                                 [self.gamma, -self.gamma, self.gamma, -self.gamma]]))


    def update(self, t, state, flat_output):
        """
        This function receives the current time, true state, and desired flat
        outputs. It returns the command inputs.

        Inputs:
            t, present time in seconds
            state, a dict describing the present state with keys
                x, position, m
                v, linear velocity, m/s
                q, quaternion [i,j,k,w]
                w, angular velocity, rad/s
            flat_output, a dict describing the present desired flat outputs with keys
                x,        position, m
                x_dot,    velocity, m/s
                x_ddot,   acceleration, m/s**2
                x_dddot,  jerk, m/s**3
                x_ddddot, snap, m/s**4
                yaw,      yaw angle, rad
                yaw_dot,  yaw rate, rad/s

        Outputs:
            control_input, a dict describing the present computed control inputs with keys
                cmd_motor_speeds, rad/s
                cmd_thrust, N (for debugging and laboratory; not used by simulator)
                cmd_moment, N*m (for debugging; not used by simulator)
                cmd_q, quaternion [i,j,k,w] (for laboratory; not used by simulator)
        """
        cmd_motor_speeds = np.zeros((4,))
        cmd_thrust = 0
        cmd_moment = np.zeros((3,))
        cmd_q = np.zeros((4,))

        # STUDENT CODE HERE'
        ev = state['v'] - flat_output['x_dot']
        ep = state['x'] - flat_output['x']

        # if np.linalg.norm(ep) <= 1:
        #     Kp = self.Kp_agg
        #     Kd = self.Kd_agg
        # else:
        Kp = self.Kp
        Kd = self.Kd

        r_ddot = flat_output['x_ddot'] - Kd @ ev - Kp @ ep
        # r_ddot = flat_output['x_ddot'] -self.Kd @ (state['v']-flat_output['x_dot']) - self.Kp @ (state['x']-flat_output['x'])
        Fdes = self.mass * r_ddot + self.weight_vector
        b3des = Fdes / np.linalg.norm(Fdes)
        a_yaw = np.array([np.cos(flat_output['yaw']), np.sin(flat_output['yaw']), 0])
        b2des = np.cross(b3des, a_yaw)
        b2des = b2des / np.linalg.norm(b2des)
        Rdes = np.c_[np.cross(b2des, b3des), b2des, b3des]
        R = Rotation.from_quat(state['q']).as_matrix()
        eR = 0.5 * (Rdes.T @ R - R.T @ Rdes)
        eR = np.array([eR[2,1], eR[0,2], eR[1,0]])
        #TODO: Figure out what w_des is
        wdes = np.zeros(3)
        eW = state['w'] - wdes
        u1 = R[:, 2]@Fdes
        u2 = self.inertia @ (-self.Kr@eR - self.Kw@eW)
        u = self.motor_speed_calc_matrix@np.hstack([u1, u2]).reshape(4,1)
        u = np.clip(u, 0, None)

        cmd_motor_speeds = np.sqrt(u)
        cmd_motor_speeds = np.clip(cmd_motor_speeds, self.rotor_speed_min, self.rotor_speed_max)
        cmd_thrust = u1
        cmd_moment = u2
        


        control_input = {'cmd_motor_speeds':cmd_motor_speeds,
                         'cmd_thrust':cmd_thrust,
                         'cmd_moment':cmd_moment,
                         'cmd_q':cmd_q}
        return control_input
