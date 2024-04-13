from heapq import heappush, heappop  # Recommended.
import numpy as np
import itertools
from flightsim.world import World
from .occupancy_map import OccupancyMap # Recommended.
from dataclasses import dataclass
from copy import deepcopy
from collections import defaultdict
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
    print("Original resolution: ", resolution)
    original_map = OccupancyMap(world, resolution, margin)
    original_resolution = deepcopy(resolution)
    resolution_multiplier = 0.5
    threshold = 1.25
    if not np.any(np.array(original_map.map.shape) ==1):
        resolution = (resolution.max()/resolution_multiplier)*np.ones(3)
        resolution[np.array(original_map.map.shape) ==1] = original_map.resolution[np.array(original_map.map.shape) ==1]
    # resolution = np.array([0.1, 0.1, 5])
    tries = 0
    occ_map = OccupancyMap(world, resolution, margin)
    # Retrieve the index in the occupancy grid matrix corresponding to a position in space.
    start_index = tuple(occ_map.metric_to_index(start))
    goal_index = tuple(occ_map.metric_to_index(goal))
    Planner = GraphSearchPathPlanner(start_index, goal_index, occ_map, astar)
    Planner.search()

    tries += 1
    print(f"Try number: {tries}, resolution: {resolution}, Expansion: {len(Planner.closed_set)}, Linalg calls: {Planner.linalg_calls}, Neighbour calls: {Planner.neighbour_calls}")
    
    while (not Planner.Path or np.linalg.norm(np.array(Planner.Path[-1]) - np.array(goal)) > threshold*np.linalg.norm(resolution)) and tries<=3:
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
        
    if not Planner.Path or np.linalg.norm(np.array(Planner.Path[-1]) - np.array(goal)) > threshold*np.linalg.norm(resolution):
        return None, len(Planner.closed_set)
    else:
        min_dis_idx = np.argmin(np.linalg.norm(np.array(Planner.Path) - np.array(goal), axis = 1))
        Planner.Path = [np.array(start)] + Planner.Path[:min_dis_idx] + [np.array(goal)]
    
    return np.array(Planner.Path), len(Planner.closed_set)


class GraphSearchPathPlanner():
    def __init__(self, start_idx, goal_idx, occ_map, astar):
        self.start_idx = tuple(start_idx)
        self.goal_idx = tuple(goal_idx)
        self.costs = defaultdict(lambda: float('inf'))
        self.parent = {}
        self.occ_map = occ_map
        self.astar = astar
        self.neighbour_indeces = np.array(list(itertools.product([-1, 0, 1], repeat=3)), dtype=int)
        self.neighbour_indeces = np.delete(self.neighbour_indeces, 13, axis=0)
        self.current_node = None
        self.priority_queue = []
        self.closed_set = set()
        self.Path = []
        self.neighbour_calls = 0
        self.open_dict_index = 0
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

                
                
                