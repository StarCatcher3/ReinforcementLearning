from environments.src.lineworld import LineWorld
from algorithms.src.dynamic_programming import DynamicProgramming
import numpy as np

alg = DynamicProgramming()
env = LineWorld()

print(alg.value_iteration(env, np.array([[0, 1], [0, 1], [.25, .75], [.8, .2], [0, 1]])))