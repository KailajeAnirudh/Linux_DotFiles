import numpy as np
from scipy.spatial.transform import Rotation
import heapq
import numpy as np
from scipy.spatial import Rectangle
from scipy.spatial.transform import Rotation

from flightsim.world import World
from flightsim import shapes

from heapq import heappush, heappop  # Recommended.
import numpy as np
import itertools
from flightsim.world import World

from dataclasses import dataclass
from copy import deepcopy
from collections import defaultdict

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
        self.Kr = np.diag([2800, 2800, 150.0])*1.4
        self.Kw = np.diag([125, 125, 80.0])*1.4

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

######################################################################################################################


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

######################################################################################################################

def graph_search(world, resolution, margin, start, goal, astar):
    """
    Parameters:
        world,      World object representing the environment obstacles
        resolution, xyz resolution in meters for an occupancy map, shape=(3,)
        margin,     minimum allowed distance in meters from path to obstacles.
        start,      xyz position in meters, shape=(3,)
        goal,       xyz position in meters, shape=(3,)
        astar,      if True use A*, else use Dijkstra
    Output:
        return a tuple (path, nodes_expanded)
        path,       xyz position coordinates along the path in meters with
                    shape=(N,3). These are typically the centers of visited
                    voxels of an occupancy map. The first point must be the
                    start and the last point must be the goal. If no path
                    exists, return None.
        nodes_expanded, the number of nodes that have been expanded
    """

    # While not required, we have provided an occupancy map you may use or modify.
    
    original_map = OccupancyMap(world, resolution, margin)
    
    original_resolution = deepcopy(resolution)
    occupancy = original_map.map.sum()/np.ones_like(original_map.map).sum()
    map_size = np.prod(original_map.map.shape)

    print(f"Original resolution: {resolution}, Map Shape: {map_size}, Occupancy: {occupancy}")
    resolution_multiplier = 0.8
    error_threshold = 1.25
    
    # res_lev = 0.8 if map_size>32000 else 0.25
    # resolution = np.array([max(res_lev, original_resolution[0]), max(res_lev, original_resolution[1]), max(0.7, original_resolution[2])])
    # resolution[np.array(original_map.map.shape) ==1] = original_map.resolution[np.array(original_map.map.shape) ==1]

    tries = 0
    occ_map = OccupancyMap(world, resolution, margin*1.025)
    # Retrieve the index in the occupancy grid matrix corresponding to a position in space.
    start_index = tuple(occ_map.metric_to_index(start))
    goal_index = tuple(occ_map.metric_to_index(goal))
    Planner = GraphSearchPathPlanner(start_index, goal_index, occ_map, astar)
    Planner.search()

    tries += 1
    print(f"Try number: {tries}, resolution: {resolution}, Expansion: {len(Planner.closed_set)}, Linalg calls: {Planner.linalg_calls}, Neighbour calls: {Planner.neighbour_calls}")
    
    while (not Planner.Path or np.linalg.norm(np.array(Planner.Path[-1]) - np.array(goal)) > error_threshold*np.linalg.norm(resolution)) and tries<=3:
        resolution_multiplier *= 0.7
        resolution = resolution_multiplier*np.ones(3)
        resolution[np.array(original_map.map.shape) ==1] = original_map.resolution[np.array(original_map.map.shape) ==1]
        occ_map = OccupancyMap(world, resolution, margin)
        del Planner
        start_index = tuple(occ_map.metric_to_index(start)); goal_index = tuple(occ_map.metric_to_index(goal))
        Planner = GraphSearchPathPlanner(start_index, goal_index, occ_map, astar)
        Planner.search()

        print(f"Try number: {tries}, resolution: {resolution}, Expansion: {len(Planner.closed_set)}")
        tries += 1

        if resolution_multiplier < 0.07 or resolution.min()<resolution_multiplier*original_resolution.min():
            return None, len(Planner.closed_set)
        
    if not Planner.Path or np.linalg.norm(np.array(Planner.Path[-1]) - np.array(goal)) > error_threshold*np.linalg.norm(resolution):
        return None, len(Planner.closed_set)
    else:
        min_dis_idx = np.argmin(np.linalg.norm(np.array(Planner.Path) - np.array(goal), axis = 1))
        Planner.Path = [np.array(start)] + Planner.Path[:min_dis_idx] + [np.array(goal)]
    
    return np.array(Planner.Path), len(Planner.closed_set)


class GraphSearchPathPlanner():
    def __init__(self, start_idx, goal_idx, occ_map, astar):
        self.start_idx = tuple(start_idx); self.goal_idx = tuple(goal_idx)
        self.costs = defaultdict(lambda: float('inf')); self.parent = {}
        self.occ_map = occ_map; self.astar = astar
        self.neighbour_indeces = np.array(list(itertools.product([-1, 0, 1], repeat=3)), dtype=int)
        self.neighbour_indeces = np.delete(self.neighbour_indeces, 13, axis=0)
        self.current_node = None
        self.priority_queue = []; self.Path = []
        self.closed_set = set()
        self.neighbour_calls = 0; self.open_dict_index = 0
        self.linalg_calls = 0
        
        heappush(self.priority_queue, (0, self.start_idx))
        self.costs[self.start_idx] = 0
        
    def get_valid_neighbours(self):
        self.neighbour_calls += 1
        current_neighbour_idx = np.array(self.current_node, dtype=int) + self.neighbour_indeces
        # Remove neighbours that are outside the map
        current_neighbour_idx = current_neighbour_idx[((current_neighbour_idx>=0) & (current_neighbour_idx<self.occ_map.map.shape)).all(axis = 1)]
        # Remove neighbours that are occupied
        current_neighbour_idx = current_neighbour_idx[self.occ_map.map[current_neighbour_idx[:, 0], current_neighbour_idx[:, 1], current_neighbour_idx[:, 2]] == False]
        return tuple(map(tuple,current_neighbour_idx))
    
    def search(self):
        while self.priority_queue:
            current_cost, self.current_node = heappop(self.priority_queue)
            
            if self.current_node in self.closed_set:
                continue

            if self.current_node == self.goal_idx:
                self.Path = [self.occ_map.index_to_metric_center(np.array(self.goal_idx))]
                parent = self.parent[self.goal_idx]
                while parent != self.start_idx:
                    self.Path.append(self.occ_map.index_to_metric_center(np.array(parent)))
                    parent = self.parent[parent]
                self.Path.append(self.occ_map.index_to_metric_center(np.array(self.start_idx)))
                self.Path.reverse()
                return
            
            self.closed_set.add(self.current_node)
            valid_neighbours = self.get_valid_neighbours()
            
            for neighbour in valid_neighbours:    
                segment_cost =  ((neighbour[0]-self.current_node[0])**2 
                                + (neighbour[1]-self.current_node[1])**2 
                                + (neighbour[2]-self.current_node[2])**2)**0.5
                segment_cost += self.costs[self.current_node]
                
                if segment_cost < self.costs[neighbour]:
                    self.costs[neighbour] = segment_cost
                    self.parent[neighbour] = self.current_node
                    if self.astar:
                        total_cost = segment_cost + ( (neighbour[0]-self.goal_idx[0])**2 
                                                    + (neighbour[1]-self.goal_idx[1])**2 
                                                    + (neighbour[2]-self.goal_idx[2])**2)**0.5
                    else:
                        total_cost = segment_cost
                    heappush(self.priority_queue, (total_cost, neighbour))       

######################################################################################################################

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
        self.points = start.reshape(-1, 3) # shape=(n_pts,3)
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

        self.velocity = 3.25
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

######################################################################################################################