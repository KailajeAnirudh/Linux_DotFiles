import numpy as np
from heapq import heappop, heappush
from itertools import combinations, permutations
from collections import defaultdict

"""Approach: 
    -  Create a path_map
    - Find explorable cells
    - Take any combination of two exploreable cells (Start and End Points)
    - Checkpoints become waypoints, so if k ckeckpoints are there,
        Start, k! (permutations of waypoints), End
        between each waypoint find shortest path, collect path length 
    - Find Expected value.

"""
def get_neighbours(position, path_map):
    relative_movements = np.array([[0,1], [0,-1],
                                   [1,0], [-1,0]])
    neighbours = neighbours[neighbours>0]
    neighbours = neighbours[np.all(neighbours>0, axis = 1)]
    neighbours = neighbours[np.all(neighbours<path_map.shape, axis=1)]
    return neighbours

def expected_length(field, K):
    path_map = np.zeros((len(field), len(field[0])))
    
    for i, line in enumerate(field):
        path_map[i] = np.array([float(c) for c in line.replace('.', '1').replace('#','3').replace('*', '2')])
    
    explorable_areas = np.array(np.where(path_map!=3)).T
    Start_end_combinations = combinations(tuple(explorable_areas), 2)
    checkpoints = np.array(np.where(path_map==2)).T
    # checkpoint_permutations = permutations(tuple([tuple(ck) for ck in checkpoints]))

    for start_end_combo in Start_end_combinations:
        checkpoint_permutations = permutations(tuple([tuple(ck) for ck in checkpoints]))
        for checkpoint_order in checkpoint_permutations:
            path = [tuple(start_end_combo[0])]+list(checkpoint_order)+[tuple(start_end_combo[1])]
            for i in range(len(path)-2):
                segment_start = path[i]
                segment_end = path[i+1]
                priorityq = []
                g_matrix = np.zeros_like(path_map) #Matrix to keep track of costs
                g_matrix[path_map == 3] = np.inf
                parents = {}
                heappush(priorityq, (0, tuple(segment_start)))

                while priorityq:
                    u = heappop(priorityq)

                    neighbours = get_neighbours(u[1], path_map)




            


    print("Checking")

field = [
 "*#..#",
 ".#*#.",
 "*...*"]

expected_length(field, 2)