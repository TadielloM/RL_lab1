import numpy as np
import maze1 as mz 
import pandas as pd
# Description of the maze as a numpy array
maze = np.array([
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 1, 1, 1],
    [0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 1, 2, 0, 0]
])
# with the convention 
# 0 = empty cell
# 1 = obstacle
# 2 = exit of the Maze
mz.draw_maze(maze)
# Create an environment maze
env = mz.Maze(maze)
env.show()

horizon = 20

V, policy = mz.dynamic_programming(env, horizon)

method = 'DynProg'
start = (0,0,6,5)
path = env.simulate(start,policy, method)
mz.animate_solution(maze,path)
