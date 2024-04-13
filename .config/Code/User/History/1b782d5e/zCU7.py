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
        self.line_segments = np.diff(points, axis=0)
        self.segment_distances = np.linalg.norm(self.line_segments, axis=1)
        self.segment_directions = self.line_segments / self.segment_distances[:, np.newaxis]
        self.segment_times = self.segment_distances / self.velocity
        self.cumulative_times = np.cumsum(self.segment_times)
        assert self.cumulative_times.shape == (self.points.shape[0] - 1, )


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
        poly_idx = self.cumulative_times.searchsorted(t) -1 # find the index of the segment that the time t is in
        
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
    poly_idx = np.array([wp.cumulative_times.searchsorted(time) -1 for time in t])
    x = np.array([output['x'][0] for output in [wp.update(time) for time in t]])
    y = np.array([output['x'][1] for output in [wp.update(time) for time in t]])
    z = np.array([output['x'][2] for output in [wp.update(time) for time in t]])
    x_dot = np.array([output['x_dot'][0] for output in [wp.update(time) for time in t]])
    y_dot = np.array([output['x_dot'][1] for output in [wp.update(time) for time in t]])
    z_dot = np.array([output['x_dot'][2] for output in [wp.update(time) for time in t]])
    import matplotlib.pyplot as plt
    fig = plt.figure()
ax1 = fig.add_subplot(7, 1, 1, sharex=True)
ax1.plot(t, x)
ax1.set_ylabel('x')

ax2 = fig.add_subplot(7, 1, 2, sharex=True)
ax2.plot(t, y)
ax2.set_ylabel('y')

ax3 = fig.add_subplot(7, 1, 3, sharex=True)
ax3.plot(t, z)
ax3.set_ylabel('z')

ax4 = fig.add_subplot(7, 1, 4, sharex=True)
ax4.plot(t, x_dot)
ax4.set_ylabel('x_dot')

ax5 = fig.add_subplot(7, 1, 5, sharex=True)
ax5.plot(t, y_dot)
ax5.set_ylabel('y_dot')

ax6 = fig.add_subplot(7, 1, 6, sharex=True)
ax6.plot(t, z_dot)
ax6.set_ylabel('z_dot')

ax7 = fig.add_subplot(7, 1, 7, sharex=True)
ax7.plot(t, poly_idx)
ax7.set_ylabel('poly_idx')
ax7.set_xlabel('t')

plt.show()

ax1 = fig.add_subplot(6, 1, 1, sharex=True)
ax1.plot(t, x)
ax1.set_ylabel('x')

ax2 = fig.add_subplot(6, 1, 2, sharex=True)
ax2.plot(t, y)
ax2.set_ylabel('y')

ax3 = fig.add_subplot(6, 1, 3, sharex=True)
ax3.plot(t, z)
ax3.set_ylabel('z')

ax4 = fig.add_subplot(6, 1, 4, sharex=True)
ax4.plot(t, x_dot)
ax4.set_ylabel('x_dot')

ax5 = fig.add_subplot(6, 1, 5, sharex=True)
ax5.plot(t, y_dot)
ax5.set_ylabel('y_dot')

ax6 = fig.add_subplot(6, 1, 6, sharex=True)
ax6.plot(t, z_dot)
ax6.set_ylabel('z_dot')
ax6.set_xlabel('t')

plt.show()

    ax1 = fig.add_subplot(7, 1, 1)
    ax1.plot(t, x)
    ax1.set_ylabel('x')

    ax2 = fig.add_subplot(7, 1, 2)
    ax2.plot(t, y)
    ax2.set_ylabel('y')

    ax3 = fig.add_subplot(7, 1, 3)
    ax3.plot(t, z)
    ax3.set_ylabel('z')

    ax4 = fig.add_subplot(7, 1, 4)
    ax4.plot(t, x_dot)
    ax4.set_ylabel('x_dot')

    ax5 = fig.add_subplot(7, 1, 5)
    ax5.plot(t, y_dot)
    ax5.set_ylabel('y_dot')

    ax6 = fig.add_subplot(7, 1, 6)
    ax6.plot(t, z_dot)
    ax6.set_ylabel('z_dot')

    ax7 = fig.add_subplot(7, 1, 7)
    ax7.plot(t, poly_idx)
    ax7.set_ylabel('poly_idx')
    ax7.set_xlabel('t')

    plt.show()
    ax1 = fig.add_subplot(6, 1, 1)
    ax1.plot(t, x)
    ax1.set_ylabel('x')

    ax2 = fig.add_subplot(6, 1, 2)
    ax2.plot(t, y)
    ax2.set_ylabel('y')

    ax3 = fig.add_subplot(6, 1, 3)
    ax3.plot(t, z)
    ax3.set_ylabel('z')

    ax4 = fig.add_subplot(6, 1, 4)
    ax4.plot(t, x_dot)
    ax4.set_ylabel('x_dot')

    ax5 = fig.add_subplot(6, 1, 5)
    ax5.plot(t, y_dot)
    ax5.set_ylabel('y_dot')

    ax6 = fig.add_subplot(6, 1, 6)
    ax6.plot(t, z_dot)
    ax6.set_ylabel('z_dot')
    ax6.set_xlabel('t')

    plt.show()