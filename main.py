from environments.src.lineworld import LineWorld
from environments.src.gridworld import GridWorld
from environments.src.rockpaperscissors import RockPaperScissors
from environments.src.secret_envs_wrapper import SecretEnv0
from algorithms.src.dynamic_programming import DynamicProgramming
import numpy as np

alg = DynamicProgramming()
#env = SecretEnv0()
env = GridWorld()
#env = LineWorld()

print(env.num_actions())
print(env.num_states())
#pi = np.array([[0, 1, 0]] * 8192)
#pi = np.array([[0, .1, .45, .45]] * 25)
#pi = np.array([[.2, .5, .3]] * 8)
pi = np.array([[1, 0, 0]] * 8)
#pi = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [.5, .5, 0], [0, 0, 0]])
#print(env.is_forbidden(1))
print(alg.policy_iteration(env))

