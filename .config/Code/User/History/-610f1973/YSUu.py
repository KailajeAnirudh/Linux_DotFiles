import inspect
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import time
import os
from flightsim.axes3ds import Axes3Ds
from flightsim.world import World

from occupancy_map import OccupancyMap
from graph_search import graph_search

# Choose a test example file. You should write your own example files too!
# filename = 'test_empty.json'
filenames = ['test_custom.json']
astar_condition = [True, False]
resolutions = [np.ones(3)*0.5, np.ones(3)*0.25, np.ones(3)*0.1, np.ones(3)*0.05]
total_time = 0
for filename in filenames:
    # Load the test example.
    print(f'\n\nRunning {filename}')
    file = Path(inspect.getsourcefile(lambda:0)).parent.resolve() / '..' / 'util' / filename
    world = World.from_file(file)          # World boundary and obstacles.
    for resolution in resolutions: # (x,y,z) resolution of discretization, shape=(3,).
        margin = world.world['margin']         # Scalar spherical robot size or safety margin.
        start  = world.world['start']          # Start point, shape=(3,)
        goal   = world.world['goal']           # Goal point, shape=(3,)
        for astar in astar_condition:
            # Run your code and return the path.
            start_time = time.time()
            path, node_expanded = graph_search(world, resolution, margin, start, goal, astar=True)
            end_time = time.time()

            # Print results.
            print()
            print('Total number of nodes expanded:', node_expanded)
            print(f'Solved in {end_time-start_time:.2f} seconds')
            if path is not None:
                number_points = path.shape[0]
                length = np.sum(np.linalg.norm(np.diff(path, axis=0),axis=1))
                print(f'The discrete path has {number_points} points.')
                print(f'The path length is {length:.2f} meters.')
            else:
                print('No path found.')

            # Draw the world, start, and goal.
            fig = plt.figure()
            ax = Axes3Ds(fig)
            world.draw(ax)
            ax.plot([start[0]], [start[1]], [start[2]], 'go', markersize=10, markeredgewidth=3, markerfacecolor='none')
            ax.plot( [goal[0]],  [goal[1]],  [goal[2]], 'ro', markersize=10, markeredgewidth=3, markerfacecolor='none')

            # Plot your path on the true World.
            if path is not None:
                world.draw_line(ax, path, color='blue')
                world.draw_points(ax, path, color='blue')
            ax.text(goal[0]-0.5, goal[1]-0.5, goal[2]-0.5, f'A* = {astar}', color='green')
            ax.text(start[0]-0.5, start[1]-0.5, start[2]-0.5, f'Expanded Nodes: {node_expanded}', color='green')
            ax.text(start[0]-0.5, start[1]-0.5, start[2]-0.1, f'Solve Time: {end_time-start_time:.2f}s', color='green')
            if path is not None:
                ax.text(start[0]-0.5, start[1]-0.5, start[2]-1.5, f'Path length: {length:.2f}m', color='green')

        total_time += end_time-start_time

print(f'Total time: {total_time:.2f}s')
plt.show()
print("Done")

"""
For debugging, you can visualize the path on the provided occupancy map.
Very, very slow! Comment this out if you're not using it.
fig = plt.figure()
ax = Axes3Ds(fig)
oc = OccupancyMap(world, resolution, margin)
oc.draw(ax)
ax.plot([start[0]], [start[1]], [start[2]], 'go', markersize=10, markeredgewidth=3, markerfacecolor='none')
ax.plot( [goal[0]],  [goal[1]],  [goal[2]], 'ro', markersize=10, markeredgewidth=3, markerfacecolor='none')
if path is not None:
    world.draw_line(ax, path, color='blue')
    world.draw_points(ax, path, color='blue')
"""
