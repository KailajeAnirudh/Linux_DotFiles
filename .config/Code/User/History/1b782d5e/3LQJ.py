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

        # STUDENT CODE HERE
        self.points = points
        self.velocity = 2.0
        self.points = np.insert(self.points, 0, points[0], axis=0)
        self.line_segments = np.diff(points, axis=0)
        self.segment_distances = np.linalg.norm(self.line_segments, axis=1)
        self.segment_directions = self.line_segments / self.segment_distances[:, np.newaxis]
        self.segment_times = self.segment_distances / self.velocity
        self.cumulative_times = np.cumsum(self.segment_times)
        assert self.cumulative_times.shape == (self.points.shape[0]-1, )


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
        x        = np.zeros((3,))
        x_dot    = np.zeros((3,))
        x_ddot   = np.zeros((3,))
        x_dddot  = np.zeros((3,))
        x_ddddot = np.zeros((3,))
        yaw = 0
        yaw_dot = 0

        # STUDENT CODE HERE
        poly_idx = self.cumulative_times.searchsorted(t) # find the index of the segment that the time t is in
        
        if poly_idx > self.points.shape[0] - 1:
            poly_idx = self.points.shape[0]-1

        if poly_idx < 0:
            poly_idx = 0

        if self.points.shape[0] != 1:
            if t < self.cumulative_times[0]:
                dt = t
            else:
                dt = t - self.cumulative_times[poly_idx]
            x = self.points[poly_idx] + self.segment_directions[poly_idx] * self.velocity * dt
            x_dot = self.segment_directions[poly_idx] * self.velocity

            if t>self.cumulative_times[-1]:
                x = self.points[-1]
                x_dot = np.zeros(3)
        else:
            x = self.points[0]
            x_dot = np.zeros(3)
            


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
    t = np.arange(0, 20, 1/500)
    poly_idx_array = np.array([])

    for dt in t:
        poly_idx = wp.cumulative_times.searchsorted(dt)# find the index of the segment that the time t is in
            
        if poly_idx > wp.points.shape[0] - 1:
            poly_idx = wp.points.shape[0]-1

        if poly_idx < 0:
            poly_idx = 0

        poly_idx_array = np.append(poly_idx_array, poly_idx)
    poly_idx = poly_idx_array
    import matplotlib.pyplot as plt

    x = np.array([output['x'][0] for output in [wp.update(time) for time in t]])
    y = np.array([output['x'][1] for output in [wp.update(time) for time in t]])
    z = np.array([output['x'][2] for output in [wp.update(time) for time in t]])
    x_dot = np.array([output['x_dot'][0] for output in [wp.update(time) for time in t]])
    y_dot = np.array([output['x_dot'][1] for output in [wp.update(time) for time in t]])
    z_dot = np.array([output['x_dot'][2] for output in [wp.update(time) for time in t]])

    fig, axs = plt.subplots(7, 1, figsize=(10, 20), sharex=True)

    axs[0].plot(t, x)
    axs[0].set_ylabel('x')
    axs[0].set_xlabel('Time')

    axs[1].plot(t, y)
    axs[1].set_ylabel('y')
    axs[1].set_xlabel('Time')

    axs[2].plot(t, z)
    axs[2].set_ylabel('z')
    axs[2].set_xlabel('Time')

    axs[3].plot(t, x_dot)
    axs[3].set_ylabel('x_dot')
    axs[3].set_xlabel('Time')

    axs[4].plot(t, y_dot)
    axs[4].set_ylabel('y_dot')
    axs[4].set_xlabel('Time')

    axs[5].plot(t, z_dot)
    axs[5].set_ylabel('z_dot')
    axs[5].set_xlabel('Time')

    axs[6].plot(t, poly_idx)
    axs[6].set_ylabel('poly_idx')
    axs[6].set_xlabel('Time')

    plt.tight_layout()
    plt.show()