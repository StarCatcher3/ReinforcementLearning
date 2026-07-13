from environments.src.baseenv import EnvTemplate
import numpy as np
import pandas as pd

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
                    #prob_total = 0.0
                    for s_p_index in range(S):
                        for r_index in range(R):
                            #print(f"Adding {env.p(s_index, a_index, s_p_index, r_index)} * {(env.reward(r_index) + gamma * V[s_p_index])} to total for s: {s_index}, a: {a_index}, s_p: {s_p_index}, r: {r_index}")
                            action_total += env.p(s_index, a_index, s_p_index, r_index) * (env.reward(r_index) + gamma * V[s_p_index])
                            #prob_total += env.p(s_index, a_index, s_p_index, r_index)
                    #print(f"Adding {pi[s_index, a_index]} * {action_total} to total, Prob_total = {prob_total}")
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
                    #print(f"Testing action {a}")
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
                    #print(f"New Policy {pi}")
            if policy_stable:
                break
        
        return pi, V
    
    def value_iteration(self, env: EnvTemplate, gamma: float = 0.99999, theta: float = 0.0001):
        S = env.num_states()
        A = env.num_actions()
        R = env.num_rewards()
        V = np.zeros(S)
        pi = np.zeros((S, A))

        while True:
            delta = 0
            #print("Starting Loop through all states")
            for s_index in range(S):
                v = V[s_index]
                best_a_total = 0.0
                best_a_index = None
                for a_index in range(A):
                    #print(f"Testing action {a}")
                    action_total = 0.0
                    for s_p_index in range(S):
                        for r_index in range(R):
                            action_total += env.p(s_index, a_index, s_p_index, r_index) * (env.reward(r_index) + gamma * V[s_p_index])
                    if action_total > best_a_total:
                        best_a_total = action_total
                        best_a_index = a_index
                V[s_index] = best_a_total
                if best_a_index is not None:
                    pi[s_index, :] = 0.0
                    pi[s_index, best_a_index] = 1.0
                #print(f"Updating Total : {V}")
                delta = np.maximum(delta, np.abs((best_a_total - v)))
            if delta < theta:
                break

        return pi, V
    
    def get_possible_state_changes(self, env: EnvTemplate) -> pd.DataFrame:
        S = env.num_states()
        A = env.num_actions()
        R = env.num_rewards()

        possible_state_changes = pd.DataFrame({
            "s_index": pd.Series(dtype="int64"),
            "a_index": pd.Series(dtype="int64"),
            "s_p_index": pd.Series(dtype="int64"),
            "r": pd.Series(dtype="float64"),
            "p": pd.Series(dtype="float64"),
        })
        for s_index in range(S):
            for a_index in range(A):
                for s_p_index in range(S):
                    for r_index in range(R):
                        p = env.p(s_index, a_index, s_p_index, r_index)
                        if p != 0:
                            possible_state_changes.loc[len(possible_state_changes)] = {
                                    "s_index": s_index,
                                    "a_index": a_index,
                                    "s_p_index": s_p_index,
                                    "r": env.reward(r_index),
                                    "p": p,
                                }
        
        return possible_state_changes
    
    def get_action_totals(self, env: EnvTemplate, pi: pd.DataFrame, possible_state_changes: pd.DataFrame, gamma: float = 0.99999, theta: float = 0.0001):
        S = env.num_states()
        V_df = pd.DataFrame(np.array(list(zip(np.arange(S), np.zeros(S))), dtype=[("s_index", int),("V", float)])).set_index("s_index")

        S_valid = possible_state_changes["s_index"].unique()

        possible_state_changes_with_value = possible_state_changes.merge(V_df, left_on="s_p_index", right_on="s_index", how="left")
        possible_state_changes_with_value["a_value"] = possible_state_changes_with_value["p"] * (possible_state_changes_with_value["r"] + gamma * possible_state_changes_with_value["V"])
        #print(f"Possible changes: {possible_state_changes_with_value}")
        action_totals = possible_state_changes_with_value[["s_index", "a_index", "a_value"]].groupby(["s_index","a_index"]).sum()

        return action_totals
    
    def efficient_policy_evaluation(self, env: EnvTemplate, pi_df: pd.DataFrame, possible_state_changes: pd.DataFrame, gamma: float = 0.99999, theta: float = 0.0001):
        S = env.num_states()
        V_df = pd.DataFrame(np.array(list(zip(np.arange(S), np.zeros(S))), dtype=[("s_index", int),("V", float)])).set_index("s_index")

        S_valid = possible_state_changes["s_index"].unique()

        while True:
            possible_state_changes_with_value = possible_state_changes.merge(V_df, left_on="s_p_index", right_on="s_index", how="left")
            possible_state_changes_with_value["a_value"] = possible_state_changes_with_value["p"] * (possible_state_changes_with_value["r"] + gamma * possible_state_changes_with_value["V"])
            #print(f"Possible changes: {possible_state_changes_with_value}")
            action_totals = possible_state_changes_with_value[["s_index", "a_index", "a_value"]].groupby(["s_index","a_index"]).sum()
            #print(f"Action Totals: {action_totals}")
            v = V_df.copy()
            #print("Starting Loop through all states")
            for s_index in S_valid:
                #print(f"Checking index: {s_index}")
                V_df.iloc[s_index] = action_totals.loc[s_index, pi_df.loc[s_index].a_index]
                #print(f"Updating Total : {V_df}")
            #print(f"Values updated, change : {v["V"]} - {V_df["V"]} = {v["V"] - V_df["V"]}, max = {(v["V"] - V_df["V"]).abs().max()}")
            if (v["V"] - V_df["V"]).abs().max() < theta:
                break

        return V_df

    def efficient_policy_iteration(self, env: EnvTemplate, gamma: float = 0.99999, theta: float = 0.0001):
        S = env.num_states()

        # Get state changes with Probability > 0.0 as DataFrame to perform column-wise operations
        possible_state_changes = self.get_possible_state_changes(env)

        S_valid = possible_state_changes["s_index"].unique()
        pi_df = pd.DataFrame({"s_index": np.arange(S), "a_index": np.zeros(S)}).set_index("s_index")

        while True:
        
            V_df = self.efficient_policy_evaluation(env, pi_df, possible_state_changes, gamma=gamma, theta=theta)
            #print(f"Policy Evaluated: {pi}, {V}")
            
            policy_stable = True

            possible_state_changes_with_value = possible_state_changes.merge(V_df, left_on="s_p_index", right_on="s_index", how="left")
            possible_state_changes_with_value["a_value"] = possible_state_changes_with_value["p"] * (possible_state_changes_with_value["r"] + gamma * possible_state_changes_with_value["V"])
            #print(f"Possible changes: {possible_state_changes_with_value}")
            action_totals = possible_state_changes_with_value[["s_index", "a_index", "a_value"]].groupby(["s_index","a_index"]).sum()
            
            best_a_df = action_totals.groupby("s_index")["a_value"].idxmax().map(lambda x: x[1])
            for s_index in S_valid:
                best_a = best_a_df.loc[s_index]
                if best_a != pi_df.loc[s_index].a_index:
                    #print(f"Not best action {best_a}, {pi_df.loc[s_index].a_index}")
                    pi_df.loc[s_index] = best_a
                    policy_stable = False
                    #print(f"New Policy {pi}")
            if policy_stable:
                break
        
        return pi_df, V_df

    def efficient_value_iteration(self, env: EnvTemplate, gamma: float = 0.99999, theta: float = 0.0001):
        S = env.num_states()
        V_df = pd.DataFrame(np.array(list(zip(np.arange(S), np.zeros(S))), dtype=[("s_index", int),("V", float)])).set_index("s_index")

        # Get state changes with Probability > 0.0 as DataFrame to perform column-wise operations
        possible_state_changes = self.get_possible_state_changes(env)

        S_valid = possible_state_changes["s_index"].unique()

        while True:
            possible_state_changes_with_value = possible_state_changes.merge(V_df, left_on="s_p_index", right_on="s_index", how="left")
            possible_state_changes_with_value["a_value"] = possible_state_changes_with_value["p"] * (possible_state_changes_with_value["r"] + gamma * possible_state_changes_with_value["V"])
            #print(f"Possible changes: {possible_state_changes_with_value}")
            action_totals = possible_state_changes_with_value[["s_index", "a_index", "a_value"]].groupby(["s_index","a_index"]).sum()
            #print(f"Action Totals: {action_totals}")
            v = V_df.copy()
            #print("Starting Loop through all states")
            for s_index in S_valid:
                #print(f"Checking index: {s_index}")
                V_df.iloc[s_index] = action_totals.loc[s_index].max()
                #print(f"Updating Total : {V_df}")
            #print(f"Values updated, change : {v["V"]} - {V_df["V"]} = {v["V"] - V_df["V"]}, max = {(v["V"] - V_df["V"]).abs().max()}")
            if (v["V"] - V_df["V"]).abs().max() < theta:
                break

        possible_state_changes_with_value = possible_state_changes.merge(V_df, left_on="s_p_index", right_on="s_index", how="left")
        possible_state_changes_with_value["a_value"] = possible_state_changes_with_value["p"] * (possible_state_changes_with_value["r"] + gamma * possible_state_changes_with_value["V"])
        action_totals = possible_state_changes_with_value[["s_index", "a_index", "a_value"]].groupby(["s_index","a_index"]).sum()
        #pi_df = action_totals.groupby(["s_index"]).max()[["s_index", "a_index"]]
        pi_df = action_totals.groupby("s_index")["a_value"].idxmax().map(lambda x: x[1]).to_frame()
        pi_df.columns = ["a_index"]
        return pi_df, V_df