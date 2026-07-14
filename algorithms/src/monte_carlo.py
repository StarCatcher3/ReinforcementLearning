from environments.src.baseenv import EnvTemplate
import numpy as np

class MonteCarlo():

    def choose_action(self, actions, pi_s: np.array) -> int:
        if len(actions) != len(pi_s):
            pi_s = [pi_s[i] for i in actions]
        return np.random.choice(actions, p=pi_s)

    def exploring_starts(self, env: EnvTemplate, iterations: int = 10_000, gamma: float = 0.99999):
        print(env.state_id())
        S = env.num_states()
        A = env.num_actions()
        Q = np.zeros((S, A))
        pi = np.zeros(S, dtype=int)
        for s_index in range(S):
            pi[s_index] = np.random.randint(0, A)
        
        #print(f"Initial Policy: {pi}")

        return_counts = np.zeros((S, A))

        i = 0
        while i < iterations:
            #print("Initializing Random State")
            env = env.from_random_state()
            #print(f"Starting State: {env.state_id()}")
            #print(f"Current Policy: {pi}")
            #print("Starting Loop through all states")
            episode = []
            visited = []
            G = 0
            while not env.is_game_over():
                s = env.state_id()
                #print(f"State: {s}")
                a = pi[s]
                #print(f"Action: {a}")
                score = env.score()
                env.step(a)
                episode.append({"S": s, "A": a, "R": env.score() - score, "First": not [s, a] in visited})
                visited.append([s, a])
                # Stop episode of stuck in a loop and penalise slightly to stop infinite loops
                if visited.count([s, a]) >= 10:
                    G = -0.000001
                    break
            
            #print(f"Episode steps: {episode}")
            for step in reversed(episode):
                G = gamma * G + step["R"]
                if step['First']:
                    s = step["S"]
                    a = step["A"]
                    Q[s, a] = (Q[s, a] * return_counts[s, a] + G) / (return_counts[s, a] + 1)
                    return_counts[s, a] += 1
                    pi = [np.argmax(a) for a in Q]
                    #print(f"First Visit step info: {step}")

            i += 1

        return pi, Q
    
    def on_policy_first_visit(self, env: EnvTemplate, iterations: int = 10_000, epsilon: float = 0.05, gamma: float = 0.99999):
        print(env.state_id())
        S = env.num_states()
        A = env.num_actions()
        Q = np.zeros((S, A))
        pi = np.zeros((S, A))
        for s_index in range(S):
            rdm_a_index = np.random.randint(0, A)
            pi[s_index, :] = epsilon / A
            pi[s_index, rdm_a_index] = 1.0 - epsilon + epsilon / A
        
        print(f"Initial Policy: {pi}")

        return_counts = np.zeros((S, A))

        i = 0
        while i < iterations:
            #print("Initializing Random State")
            env = env.from_random_state()
            #print(f"Starting State: {env.state_id()}")
            #print(f"Current Policy: {pi}")
            #print("Starting Loop through all states")
            episode = []
            visited = []
            G = 0
            while not env.is_game_over():
                s = env.state_id()
                #print(f"State: {s}")
                a = self.choose_action(env.available_actions(), pi[s])
                #print(f"Action: {a}")
                score = env.score()
                env.step(a)
                episode.append({"S": s, "A": a, "R": env.score() - score, "First": not [s, a] in visited})
                visited.append([s, a])
                # Stop episode of stuck in a loop and penalise slightly to stop infinite loops
                if visited.count([s, a]) >= 10:
                    G = -0.000001
                    break
            
            #print(f"Episode steps: {episode}")
            for step in reversed(episode):
                G = gamma * G + step["R"]
                if step['First']:
                    s = step["S"]
                    a = step["A"]
                    Q[s, a] = (Q[s, a] * return_counts[s, a] + G) / (return_counts[s, a] + 1)
                    return_counts[s, a] += 1
                    for s_index in range(S):
                        best_a_index = np.argmax(Q[s_index])
                        pi[s_index, :] = epsilon / A
                        pi[s_index, best_a_index] = 1.0 - epsilon + epsilon / A
                    #print(f"First Visit step info: {step}")

            i += 1

        return pi, Q