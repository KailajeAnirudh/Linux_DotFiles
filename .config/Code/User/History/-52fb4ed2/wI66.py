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

def expected_length(field, K):
    map = np.zeros((len(field), len(field[0])))
    
    for i, line in enumerate(field):
        map[i] = np.array([float(c) for c in line.replace('.', '1').replace('#','3').replace('*', '2')])
    
    explorable_areas = np.array(np.where(map!=3)).T
    Start_end_combinations = combinations(tuple(explorable_areas), 2)
    checkpoints = np.array(np.where(map==2)).T
    checkpoint_permutations = permutations(tuple(checkpoints))

    for start_end_combo in Start_end_combinations:
        for checkpoint_order in checkpoint_permutations:
            path = [start_end_combo[0]]+list(checkpoint_order)+[start_end_combo[1]]
            for i in range(len(path)-2):
                segment_start = path[i]
                segment_end = path[i+1]
                priorityq = []
                heappush(priorityq, (0, tuple(segment_start)))


            


    print("Checking")

field = [
 "*#..#",
 ".#*#.",
 "*...*"]

expected_length(field, 2)