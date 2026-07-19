from environnements.src.baseenv import EnvTemplate
import numpy as np


class DynamicProgramming():

    def policy_evaluation(self, env: EnvTemplate, pi: np.array, gamma: float = 0.99999, theta: float = 0.0001) -> np.array:
        S = env.maximum_states_count()
        A = env.maximum_actions_count()
        R = env.num_rewards()
        V = np.zeros(S)
        while True:
            delta = 0
            for s in range(S):
                v = V[s]
                total = 0.0
                for a in range(A):
                    action_total = 0.0
                    for s_p in range(S):
                        for r_index in range(R):
                            action_total += env.p(s, a, s_p, r_index) * (env.reward(r_index) + gamma * V[s_p])
                    total += pi[s, a] * action_total
                V[s] = total
                delta = max(delta, abs(v - V[s]))
            if delta < theta:
                break
        return V

    def policy_iteration(self, env: EnvTemplate, gamma: float = 0.99999, theta: float = 0.0001) -> tuple[np.array, np.array]:
        S = env.maximum_states_count()
        A = env.maximum_actions_count()
        R = env.num_rewards()
        V = np.zeros(S)
        pi = np.zeros(S, dtype=int)

        while True:
            while True:
                delta = 0
                for s in range(S):
                    v = V[s]
                    total = 0.0
                    for s_p in range(S):
                        for r_index in range(R):
                            total += env.p(s, pi[s], s_p, r_index) * (env.reward(r_index) + gamma * V[s_p])
                    V[s] = total
                    delta = max(delta, abs(v - V[s]))
                if delta < theta:
                    break

            policy_stable = True
            for s in range(S):
                old_action = pi[s]
                best_action = None
                best_value = -np.inf
                for a in range(A):
                    action_total = 0.0
                    for s_p in range(S):
                        for r_index in range(R):
                            action_total += env.p(s, a, s_p, r_index) * (env.reward(r_index) + gamma * V[s_p])
                    if best_action is None or action_total >= best_value:
                        best_value = action_total
                        best_action = a
                pi[s] = best_action
                if old_action != pi[s]:
                    policy_stable = False

            if policy_stable:
                return V, pi

    def value_iteration(self, env: EnvTemplate, gamma: float = 0.99999, theta: float = 0.0001) -> tuple[np.array, np.array]:
        S = env.maximum_states_count()
        A = env.maximum_actions_count()
        R = env.num_rewards()
        V = np.zeros(S)

        while True:
            delta = 0
            for s in range(S):
                v = V[s]
                best_value = -np.inf
                for a in range(A):
                    action_total = 0.0
                    for s_p in range(S):
                        for r_index in range(R):
                            action_total += env.p(s, a, s_p, r_index) * (env.reward(r_index) + gamma * V[s_p])
                    best_value = max(best_value, action_total)
                V[s] = best_value
                delta = max(delta, abs(v - V[s]))
            if delta < theta:
                break

        pi = np.zeros(S, dtype=int)
        for s in range(S):
            best_action = None
            best_value = -np.inf
            for a in range(A):
                action_total = 0.0
                for s_p in range(S):
                    for r_index in range(R):
                        action_total += env.p(s, a, s_p, r_index) * (env.reward(r_index) + gamma * V[s_p])
                if best_action is None or action_total >= best_value:
                    best_value = action_total
                    best_action = a
            pi[s] = best_action

        return V, pi
