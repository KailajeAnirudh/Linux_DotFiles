import numpy as np
from scipy.spatial.transform import Rotation
import heapq
import numpy as np
from scipy.spatial import Rectangle
from scipy.spatial.transform import Rotation

from flightsim.world import World
from flightsim import shapes


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
        # self.Kp = np.diag([7.5, 7.5, 35.0])*1.05
        # self.Kd = np.diag([4.5,4.5,12.0/1.025])/1.025
        
        # self.Kr = np.diag([11000, 11000, 2200.0])
        # self.Kw = np.diag([335, 335, 200.0])

        self.Kp = np.diag([8, 8, 19.0])#/4.2
        self.Kd = np.diag([5.5,5.5,9.0])#/2.5
        self.Kr = np.diag([2800, 2800, 150.0])
        self.Kw = np.diag([125, 125, 80.0])

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



class OccupancyMap:
    def __init__(self, world=World.empty((0, 2, 0, 2, 0, 2)), resolution=(.1, .1, .1), margin=.2):
        """
        This class creates a 3D voxel occupancy map of the configuration space from a flightsim World object.
        Parameters:
            world, a flightsim World object
            resolution, the discretization of the occupancy grid in x,y,z
            margin, the inflation radius used to create the configuration space (assuming a spherical drone)
        """
        self.world = world
        self.resolution = np.array(resolution)
        self.margin = margin
        self._init_map_from_world()

    def index_to_metric_negative_corner(self, index):
        """
        Return the metric position of the most negative corner of a voxel, given its index in the occupancy grid
        """
        return index*np.array(self.resolution) + self.origin

    def index_to_metric_center(self, index):
        """
        Return the metric position of the center of a voxel, given its index in the occupancy grid
        """
        return self.index_to_metric_negative_corner(index) + self.resolution/2.0

    def metric_to_index(self, metric):
        """
        Returns the index of the voxel containing a metric point.
        Remember that this and index_to_metric and not inverses of each other!
        If the metric point lies on a voxel boundary along some coordinate,
        the returned index is the lesser index.
        """
        return np.floor((metric - self.origin)/self.resolution).astype('int')

    def _metric_block_to_index_range(self, bounds, outer_bound=True):
        """
        A fast test that returns the closed index range intervals of voxels
        intercepting a rectangular bound. If outer_bound is true the returned
        index range is conservatively large, if outer_bound is false the index
        range is conservatively small.
        """

        # Implementation note: The original intended resolution may not be
        # exactly representable as a floating point number. For example, the
        # floating point value for "0.1" is actually bigger than 0.1. This can
        # cause surprising results on large maps. The solution used here is to
        # slightly inflate or deflate the resolution by the smallest
        # representative unit to achieve either an upper or lower bound result.
        sign = 1 if outer_bound else -1
        min_index_res = np.nextafter(self.resolution,  sign * np.inf) # Use for lower corner.
        max_index_res = np.nextafter(self.resolution, -sign * np.inf) # Use for upper corner.

        bounds = np.asarray(bounds)
        # Find minimum included index range.
        min_corner = bounds[0::2]
        min_frac_index = (min_corner - self.origin)/min_index_res
        min_index = np.floor(min_frac_index).astype('int')
        min_index[min_index == min_frac_index] -= 1
        min_index = np.maximum(0, min_index)
        # Find maximum included index range.
        max_corner = bounds[1::2]
        max_frac_index = (max_corner - self.origin)/max_index_res
        max_index = np.floor(max_frac_index).astype('int')
        max_index = np.minimum(max_index, np.asarray(self.map.shape)-1)
        return (min_index, max_index)

    def _init_map_from_world(self):
        """
        Creates the occupancy grid (self.map) as a boolean numpy array. True is
        occupied, False is unoccupied. This function is called during
        initialization of the object.
        """

        # Initialize the occupancy map, marking all free.
        bounds = self.world.world['bounds']['extents']
        voxel_dimensions_metric = []
        voxel_dimensions_indices = []
        for i in range(3):
            voxel_dimensions_metric.append(abs(bounds[1+i*2]-bounds[i*2]))
            voxel_dimensions_indices.append(int(np.ceil(voxel_dimensions_metric[i]/self.resolution[i])))
        self.map = np.zeros(voxel_dimensions_indices, dtype=bool)
        self.origin = np.array(bounds[0::2])

        # Iterate through each block obstacle.
        for block in self.world.world.get('blocks', []):
            extent = block['extents']
            block_rect = Rectangle([extent[1], extent[3], extent[5]], [extent[0], extent[2], extent[4]])
            # Get index range that is definitely occupied by this block.
            (inner_min_index, inner_max_index) = self._metric_block_to_index_range(extent, outer_bound=False)
            a, b = inner_min_index, inner_max_index
            self.map[a[0]:(b[0]+1), a[1]:(b[1]+1), a[2]:(b[2]+1)] = True
            # Get index range that is definitely not occupied by this block.
            outer_extent = extent + self.margin * np.array([-1, 1, -1, 1, -1, 1])
            (outer_min_index, outer_max_index) = self._metric_block_to_index_range(outer_extent, outer_bound=True)
            # Iterate over uncertain voxels with rect-rect distance check.
            for i in range(outer_min_index[0], outer_max_index[0]+1):
                for j in range(outer_min_index[1], outer_max_index[1]+1):
                    for k in range(outer_min_index[2], outer_max_index[2]+1):
                        # If map is not already occupied, check for collision.
                        if not self.map[i,j,k]:
                            metric_loc = self.index_to_metric_negative_corner((i,j,k))
                            voxel_rect = Rectangle(metric_loc+self.resolution, metric_loc)
                            rect_distance = voxel_rect.min_distance_rectangle(block_rect)
                            self.map[i,j,k] = rect_distance <= self.margin

    def draw_filled(self, ax):
        """
        Visualize the occupancy grid (mostly for debugging)
        Warning: may be slow with O(10^3) occupied voxels or more
        Parameters:
            ax, an Axes3D object
        """
        self.world.draw_empty_world(ax)
        it = np.nditer(self.map, flags=['multi_index'])
        while not it.finished:
            if self.map[it.multi_index] == True:
                metric_loc = self.index_to_metric_negative_corner(it.multi_index)
                xmin, ymin, zmin = metric_loc
                xmax, ymax, zmax = metric_loc + self.resolution
                c = shapes.Cuboid(ax, xmax-xmin, ymax-ymin, zmax-zmin, alpha=0.1, linewidth=1, edgecolors='k', facecolors='b')
                c.transform(position=(xmin, ymin, zmin))
            it.iternext()

    def _draw_voxel_face(self, ax, index, direction):
        # Normalized coordinates of the top face.
        face = np.array([(1,1,1), (-1,1,1), (-1,-1,1), (1,-1,1)])
        # Rotate to find normalized coordinates of target face.
        if   direction[0] != 0:
            axis = np.array([0, 1, 0]) * np.pi/2 * direction[0]
        elif direction[1] != 0:
            axis = np.array([-1, 0, 0]) * np.pi/2 * direction[1]
        elif direction[2] != 0:
            axis = np.array([1, 0, 0]) * np.pi/2 * (1-direction[2])
        face = (Rotation.from_rotvec(axis).as_matrix() @ face.T).T
        # Scale, position, and draw using Face object.
        face = 0.5 * face * np.reshape(self.resolution, (1,3))
        f = shapes.Face(ax, face, alpha=0.1, linewidth=1, edgecolors='k', facecolors='b')
        f.transform(position=(self.index_to_metric_center(index)))

    def draw_shell(self, ax):
        self.world.draw_empty_world(ax)
        it = np.nditer(self.map, flags=['multi_index'])
        while not it.finished:
            idx = it.multi_index
            if self.map[idx] == True:
                for d in [(0,0,-1), (0,0,1), (0,-1,0), (0,1,0), (-1,0,0), (1,0,0)]:
                    neigh_idx = (idx[0]+d[0], idx[1]+d[1], idx[2]+d[2])
                    neigh_exists = self.is_valid_index(neigh_idx)
                    if not neigh_exists or (neigh_exists and not self.map[neigh_idx]):
                        self._draw_voxel_face(ax, idx, d)
            it.iternext()

    def draw(self, ax):
        self.draw_shell(ax)

    def is_valid_index(self, voxel_index):
        """
        Test if a voxel index is within the map.
        Returns True if it is inside the map, False otherwise.
        """
        for i in range(3):
            if voxel_index[i] >= self.map.shape[i] or voxel_index[i] < 0:
                return False
        return True

    def is_valid_metric(self, metric):
        """
        Test if a metric point is within the world.
        Returns True if it is inside the world, False otherwise.
        """
        bounds = self.world.world['bounds']['extents']
        for i in range(3):
            if metric[i] <= bounds[i*2] or metric[i] >= bounds[i*2+1]:
                return False
        return True

    def is_occupied_index(self, voxel_index):
        """
        Test if a voxel index is occupied.
        Returns True if occupied or outside the map, False otherwise.
        """
        return (not self.is_valid_index(voxel_index)) or self.map[tuple(voxel_index)]

    def is_occupied_metric(self, voxel_metric):
        """
        Test if a metric point is within an occupied voxel.
        Returns True if occupied or outside the map, False otherwise.
        """
        ind = self.metric_to_index(voxel_metric)
        return (not self.is_valid_index(ind)) or self.is_occupied_index(ind)

