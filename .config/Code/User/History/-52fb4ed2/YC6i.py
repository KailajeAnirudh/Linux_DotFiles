import numpy as np
import heapq
from itertools import combinations
from collections import defaultdict


def expected_length(field, K):
    map = np.zeros((len(field), len(field[0])))
    priorityq = []
    for i, line in enumerate(field):
        map[i] = np.array([float(c) for c in line.replace('.', '1').replace('#','3').replace('*', '2')])
    
    explorable_areas
    print("Checking")

field = [
 "*#..#",
 ".#*#.",
 "*...*"]

expected_length(field, 2)