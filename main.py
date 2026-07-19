from environnements.src.lineworld import LineWorld
from environnements.src.gridworld import GridWorld
from environnements.src.rockpaperscissors import RockPaperScissors
from environnements.src.secret_envs_wrapper import SecretEnv0
from algorithmes.src.dynamic_programming import DynamicProgramming

import numpy as np

alg = DynamicProgramming()
env = RockPaperScissors(3)

print(env.maximum_actions_count())
print(env.maximum_states_count())
pi = np.array([[1, 0, 0]] * 8)
print(alg.policy_evaluation(env, pi))
