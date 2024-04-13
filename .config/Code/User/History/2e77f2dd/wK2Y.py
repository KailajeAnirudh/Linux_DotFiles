from heapq import heappush, heappop  # Recommended.
import numpy as np
from collections import defaultdict
from flightsim.world import World
import heapq
from .occupancy_map import OccupancyMap # Recommended.
from queue import PriorityQueue
from heapdict import heapdict

def pos_neighbors(current_pos, occ_map):
    neighbor = np.array([[1, -1, 0], [1, 0, 0], [1, 1, 0], [0, -1, 0], [0, 1, 0], [-1, -1, 0], [-1, 0, 0], [-1, 1, 0], [1, - 1, 1], [1, 0, 1],
                         [1, 1, 1],
                         [0, -1, 1],
                         [0, 1, 1],
                         [- 1, - 1, 1],
                         [-1, 0, 1],
                         [- 1, 1, 1],
                         [0, 0, 1],
                         [1, -1, -1],
                         [1, 0, -1],
                         [1, 1, -1],
                         [0, -1, -1],
                         [0, 1, -1],
                         [-1, - 1, - 1],
                         [- 1, 0, - 1],
                         [- 1, 1, - 1],
                         [0, 0, - 1]])
    neighbours = current_pos + neighbor
    valid = np.all(neighbours >= 0, axis=1) & np.all(neighbours < occ_map.map.shape, axis=1)
    valid_indices = np.where(valid)[0]
    neighbours = neighbours[valid_indices]
    x, y, z = neighbours.T
    graph_data = occ_map.map[x, y, z]
    valid_indices = np.where(graph_data == 0)[0]
    
    return neighbours[valid_indices]


def graph_search(world, resolution, margin, start, goal, astar):
    occ_map = OccupancyMap(world, resolution, margin)
    start_index = tuple(occ_map.metric_to_index(start))
    goal_index = tuple(occ_map.metric_to_index(goal))

    cell_to_visit = PriorityQueue()
    heuristic_cost_dict = defaultdict(lambda: float('inf'))
    heuristic_cost_dict[start_index] = 0
    cell_to_visit.put((0, start_index))
    child_parent_dict = {}
    explored_positions = set()

    while not cell_to_visit.empty():
        _, current_pos = cell_to_visit.get()
        current_pos = tuple(current_pos)

        if current_pos in explored_positions:
            continue

        if current_pos == goal_index:
            shortest_path = [goal]
            while current_pos != start_index:
                shortest_path.insert(0, occ_map.index_to_metric_center(current_pos))
                current_pos = child_parent_dict[current_pos]
            shortest_path.insert(0, start)
            return np.array(shortest_path), len(explored_positions)

        explored_positions.add(current_pos)

        neighbors = pos_neighbors(np.array(current_pos), occ_map)
        for next_position in neighbors:
            next_position_tuple = tuple(next_position)
            cost = np.linalg.norm(next_position - current_pos)
            new_cost = heuristic_cost_dict[current_pos] + cost
            if new_cost < heuristic_cost_dict[next_position_tuple]:
                heuristic_cost_dict[next_position_tuple] = new_cost
                priority = new_cost + np.linalg.norm(next_position - goal_index) if astar else new_cost
                cell_to_visit.put((priority, next_position_tuple))
                child_parent_dict[next_position_tuple] = current_pos

    return None, len(explored_positions)




