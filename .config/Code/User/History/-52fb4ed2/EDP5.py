import numpy as np
from heapq import heappop, heappush
from itertools import combinations, permutations
from collections import defaultdict

"""Approach: 
    -  Create a map
    - Find explorable cells
    - Take any combination of two exploreable cells (Start and End Points)
    - Checkpoints become waypoints, so if k ckeckpoints are there,
        Start, k! (permutations of waypoints), End
        between each waypoint find shortest path, collect path length 
    - Find Expected value.

"""
def get_neighbours(position):
    relative_movements = np.array([[0,1], [0,-1],
                                   [1,0], [-1,0]])
    neighbours = np.array(position)[np.newaxis, :]+relative_movements
    neighbours = neighbours[neighbours>0]
    

    return neighbours

def expected_length(field, K):
    map = np.zeros((len(field), len(field[0])))
    
    for i, line in enumerate(field):
        map[i] = np.array([float(c) for c in line.replace('.', '1').replace('#','3').replace('*', '2')])
    
    explorable_areas = np.array(np.where(map!=3)).T
    Start_end_combinations = combinations(tuple(explorable_areas), 2)
    checkpoints = np.array(np.where(map==2)).T
    # checkpoint_permutations = permutations(tuple([tuple(ck) for ck in checkpoints]))

    for start_end_combo in Start_end_combinations:
        checkpoint_permutations = permutations(tuple([tuple(ck) for ck in checkpoints]))
        for checkpoint_order in checkpoint_permutations:
            path = [tuple(start_end_combo[0])]+list(checkpoint_order)+[tuple(start_end_combo[1])]
            for i in range(len(path)-2):
                segment_start = path[i]
                segment_end = path[i+1]
                priorityq = []
                g_matrix = np.zeros_like(map) #Matrix to keep track of costs
                g_matrix[map == 3] = np.inf
                parents = {}
                heappush(priorityq, (0, tuple(segment_start)))

                while priorityq:
                    u = heappop(priorityq)

                    neighbours = get_neighbours(u[1])




            


    print("Checking")

field = [
 "*#..#",
 ".#*#.",
 "*...*"]

expected_length(field, 2)