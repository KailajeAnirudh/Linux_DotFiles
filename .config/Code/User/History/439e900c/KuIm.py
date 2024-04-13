import numpy as np

from .graph_search import graph_search

class WorldTraj(object):
    """

    """
    def __init__(self, world, start, goal):
        """
        This is the constructor for the trajectory object. A fresh trajectory
        object will be constructed before each mission. For a world trajectory,
        the input arguments are start and end positions and a world object. You
        are free to choose the path taken in any way you like.

        You should initialize parameters and pre-compute values such as
        polynomial coefficients here.

        Parameters:
            world, World object representing the environment obstacles
            start, xyz position in meters, shape=(3,)
            goal,  xyz position in meters, shape=(3,)

        """

        # You must choose resolution and margin parameters to use for path
        # planning. In the previous project these were provided to you; now you
        # must chose them for yourself. Your may try these default values, but
        # you should experiment with them!
        self.resolution = np.array([0.25, 0.25, 0.25])
        self.margin = 0.5

        # You must store the dense path returned from your Dijkstra or AStar
        # graph search algorithm as an object member. You will need it for
        # debugging, it will be used when plotting results.
        self.path, _ = graph_search(world, self.resolution, self.margin, start, goal, astar=True)

        # You must generate a sparse set of waypoints to fly between. Your
        # original Dijkstra or AStar path probably has too many points that are
        # too close together. Store these waypoints as a class member; you will
        # need it for debugging and it will be used when plotting results.
        self.points = start # shape=(n_pts,3)
        if self.path is not None:
            segment_distances = np.linalg.norm(np.diff(self.path, axis=0), axis=1)
            cumulative_distances = np.cumsum(segment_distances)
            waypoint_distances = np.linspace(0, cumulative_distances[-1], int(cumulative_distances[-1]/(0.25*1.5)))
            distances = cumulative_distances - waypoint_distances[:, np.newaxis]
            distances[distances < 0] = np.inf
            indices = np.argmin(distances, axis=1)
            self.points = self.path[indices]
            if np.all(self.points[-1] != goal):
                self.points = np.vstack((self.points, goal))

        self.velocity = 3.0
        self.line_segments = None
        self.segment_distances = None
        self.segment_directions = None
        self.segment_times = None
        self.cumulative_times = None

        if self.points.shape[0] > 1:
            self.points = np.insert(self.points, 0, self.points[0], axis=0)
            self.line_segments = np.diff(self.points, axis=0)
            self.segment_distances = np.linalg.norm(self.line_segments, axis=1)
            self.segment_directions = np.nan_to_num(self.line_segments / (self.segment_distances[:, np.newaxis]+1e-8))
            self.segment_times = self.segment_distances / self.velocity
            self.cumulative_times = np.cumsum(self.segment_times)
            self.cumulative_times = np.insert(self.cumulative_times, 0, 0)




        # Finally, you must compute a trajectory through the waypoints similar
        # to your task in the first project. One possibility is to use the
        # WaypointTraj object you already wrote in the first project. However,
        # you probably need to improve it using techniques we have learned this
        # semester.

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
        x        = np.zeros((3,))
        x_dot    = np.zeros((3,))
        x_ddot   = np.zeros((3,))
        x_dddot  = np.zeros((3,))
        x_ddddot = np.zeros((3,))
        yaw = 0
        yaw_dot = 0

        # STUDENT CODE HERE

        if self.points.shape[0] > 1:
            if t < self.cumulative_times[-1]:
                poly_idx = max(0, self.cumulative_times.searchsorted(t) - 1)
                x = self.points[poly_idx] + self.segment_directions[poly_idx] * self.velocity * (t - self.cumulative_times[poly_idx])
                x_dot = self.segment_directions[poly_idx] * self.velocity
            if t >= self.cumulative_times[-1]:
                x = self.points[-1]
                x_dot = np.zeros((3,))
        elif self.points.shape[0] == 1:
            x = self.points[0]
            x_dot = np.zeros((3,))
        else:
            x = True
            x_dot = np.zeros((3,))

        flat_output = { 'x':x, 'x_dot':x_dot, 'x_ddot':x_ddot, 'x_dddot':x_dddot, 'x_ddddot':x_ddddot,
                        'yaw':yaw, 'yaw_dot':yaw_dot}
        return flat_output
