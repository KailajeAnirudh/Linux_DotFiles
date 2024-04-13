import numpy as np

class WaypointTraj(object):
    """

    """
    def __init__(self, points):
        """
        This is the constructor for the Trajectory object. A fresh trajectory
        object will be constructed before each mission. For a waypoint
        trajectory, the input argument is an array of 3D destination
        coordinates. You are free to choose the times of arrival and the path
        taken between the points in any way you like.

        You should initialize parameters and pre-compute values such as
        polynomial coefficients here.

        Inputs:
            points, (N, 3) array of N waypoint coordinates in 3D
        """
        self.points = points
        self.velocity = 2
        # STUDENT CODE HERE

    def update(self, t):
        """
        Given the present time, return the desired flat output and derivatives.

        Inputs
            t, time, s
        Outputs
            flat_output, a dict describing the present desired flat outputs with keys
                x,        position, m
                x_dot,    velocity, m/s
                x_ddot,   acceleration, m/s**2
                x_dddot,  jerk, m/s**3
                x_ddddot, snap, m/s**4
                yaw,      yaw angle, rad
                yaw_dot,  yaw rate, rad/s
        """
        x = np.zeros((3,))
        x_dot = np.zeros((3,))
        x_ddot   = np.zeros((3,))
        x_dddot  = np.zeros((3,))
        x_ddddot = np.zeros((3,))
        yaw = 0
        yaw_dot = 0
        flat_output = { 'x':x, 'x_dot':x_dot, 'x_ddot':x_ddot, 'x_dddot':x_dddot, 'x_ddddot':x_ddddot,
                        'yaw':yaw, 'yaw_dot':yaw_dot}
        # print(self.points)
        i_cap = np.zeros((3,))
        d_i = []
        t_i = []
        r_i = np.zeros((3,))
        x = np.zeros((3,))
        x_dot = np.zeros((3,))
        t_start_i = []
        # print(np.shape(self.points)[0])
        if len(self.points)==0:
            # flat_output.add(True, 'hover')
            # flat_output['hover'] = True
            x = True
            x_dot = np.zeros((3,))
        elif np.shape(self.points)==(3,):
            x = self.points
            # print(x)
            x_dot = np.zeros((3,))
        else:
            # print("hey")
            for i in range(1, np.shape(self.points)[0]):
                i_cap_i = (self.points[i] - self.points[i-1])/np.linalg.norm((self.points[i] - self.points[i-1]))
                i_cap = np.vstack((i_cap,i_cap_i))
                d = np.linalg.norm((self.points[i] - self.points[i-1]))
                d_i = np.append(d_i,d)
                ta = d/self.velocity
                t_i = np.append(t_i,ta)
                t_start = sum(t_i)
                t_start_i = np.append(t_start_i,t_start)
            i_cap = np.delete(i_cap, 0, 0)
            t_start_i = np.hstack((np.array([0]),t_start_i))
            r_dot = self.velocity*i_cap
            for i in range(np.shape(self.points)[0]-1):
                if t<t_start_i[i+1] and t>=t_start_i[i]:
                    k = self.velocity*(t - t_start_i[i])
                    r = self.points[i] + k*i_cap[i]
                    x = r
                    x_dot = r_dot[i]
                    break
                elif t>=t_start_i[-1]:
                    k = self.velocity*(t_start_i[-1] - t_start_i[-2])
                    r = self.points[-2] + k*i_cap[-1]
                    x = r
                    x_dot = r_dot[-1]
                    x_dot = np.zeros((3,))
                    break
        x_ddot   = np.zeros((3,))
        x_dddot  = np.zeros((3,))
        x_ddddot = np.zeros((3,))
        yaw = 0
        yaw_dot = 0

        # STUDENT CODE HERE
        
        # STUDENT CODE END HERE

        flat_output = { 'x':x, 'x_dot':x_dot, 'x_ddot':x_ddot, 'x_dddot':x_dddot, 'x_ddddot':x_ddddot,
                        'yaw':yaw, 'yaw_dot':yaw_dot}
        return flat_output

if __name__ == "__main__":
    # points = np.array([[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1],[1,1,1]])
    # points = np.array([[0.0, 0.0, 1.0],[0.0, 1.0, 1.0],[1.0, 1.0, 1.0],[0.0, 0.0, 1.0]])
    points = np.array([
     [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 2.0, 0.0],
        [2.0, 2.0, 2.0],
        [0.0, 2.0, 2.0],
        [0.0, 0.0, 2.0]])
    # points = np.array([])
    wp = WaypointTraj(points)
    for t in range(3):
        flat_output = wp.update(t)
        print("flat_output",flat_output)