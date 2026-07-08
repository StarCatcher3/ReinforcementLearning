from environments.src.baseenv import EnvTemplate
import numpy as np

class DynamicProgramming():

    def policy_evaluation(self, env: EnvTemplate, pi: np.array, gamma: float = 0.99999, theta: float = 0.0001):
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
                    prob_total = 0.0
                    for s_p_index in range(S):
                        for r_index in range(R):
                            #print(f"Adding {env.p(s_index, a_index, s_p_index, r_index)} * {(env.reward(r_index) + gamma * V[s_p_index])} to total for s: {s_index}, a: {a_index}, s_p: {s_p_index}, r: {r_index}")
                            action_total += env.p(s_index, a_index, s_p_index, r_index) * (env.reward(r_index) + gamma * V[s_p_index])
                            #prob_total += env.p(s_index, a_index, s_p_index, r_index)
                    #print(f"Adding {pi[s_index, a_index]} * {action_total} to total, Prop_total = {prob_total}")
                    total += pi[s_index, a_index] * action_total
                V[s_index] = total
                #print(f"Updating Total : {V}")
                delta = np.maximum(delta, np.abs((total - v)))
            if delta < theta:
                break
        return V

    def policy_iteration(self, env: EnvTemplate, gamma: float = 0.99999, theta: float = 0.0001):
        
        S = env.num_states()
        A = env.num_actions()
        R = env.num_rewards()
        pi = np.zeros((S, A))
        for s_index in range(S):
            rdm_a_index = np.random.randint(0, A)
            pi[s_index, rdm_a_index] = 1.0

        while True:
        
            V = self.policy_evaluation(env, pi, gamma=gamma, theta=theta)
            #print(f"Policy Evaluated: {pi}, {V}")
            
            policy_stable = True
            for s_index in range(S):
                a = np.argmax(pi[s_index])
                best_a = 0.0
                best_a_index = None
                for a_index in range(A):
                    action_total = 0.0
                    for s_p_index in range(S):
                        for r_index in range(R):
                            action_total += env.p(s_index, a_index, s_p_index, r_index) * (env.reward(r_index) + gamma * V[s_p_index])
                    if action_total > best_a:
                        best_a_index = a_index
                        best_a = action_total
                if best_a_index is not None and a != best_a_index:
                    #print(f"Not best action {a}, {best_a_index}")
                    pi[s_index, :] = 0.0
                    pi[s_index, best_a_index] = 1.0
                    policy_stable = False
            if policy_stable:
                break
        
        return pi, V