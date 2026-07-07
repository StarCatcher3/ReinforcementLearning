from environments.src.baseenv import EnvTemplate
import numpy as np

class DynamicProgramming():

    def value_iteration(self, env: EnvTemplate, pi: np.array, gamma: float = 0.99999, theta: float = 0.0001):
        S = env.num_states()
        A = env.num_actions()
        R = env.num_rewards()
        V = np.zeros(S)
        while True:
            delta = 0
            for s_index in range(S):
                v = V[s_index]
                total = 0.0
                for a_index in range(A):
                    action_total = 0.0
                    for s_p_index in range(S):
                        for r_index in range(R):
                            action_total += env.p(s_index, a_index, s_p_index, r_index) * (env.reward(r_index) + gamma * V[s_p_index])
                    total += pi[s_index, a_index] * action_total
                V[s_index] = total
                delta = np.maximum(delta, np.abs((total - v)))
            if delta < theta:
                break
        return V