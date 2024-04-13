import numpy as np
import heapq
from itertools import combinations
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
    priorityq = []
    for i, line in enumerate(field):
        map[i] = np.array([float(c) for c in line.replace('.', '1').replace('#','3').replace('*', '2')])
    
    explorable_areas = np.array(np.where(map!=3)).T
    Start_end_combinations = combinations(tuple(explorable_areas), 2)
    

    print("Checking")

field = [
 "*#..#",
 ".#*#.",
 "*...*"]

expected_length(field, 2)