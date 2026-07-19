from environnements.src.baseenv import EnvTemplate
from collections import defaultdict
import numpy as np


class Planning():

    @staticmethod
    def _reward_of_step(env: EnvTemplate, action: int) -> float:
        score_before = env.score()
        env.step(action)
        return env.score() - score_before

    @staticmethod
    def _epsilon_greedy(Q: dict, s: int, available_actions, epsilon: float) -> int:
        if np.random.rand() < epsilon:
            return np.random.choice(available_actions)
        best_action = None
        best_value = 0.0
        for a in available_actions:
            total = Q[(s, a)]
            if best_action is None or total >= best_value:
                best_action = a
                best_value = total
        return best_action

    def dyna_q(self, env: EnvTemplate, alpha: float = 0.1, epsilon: float = 0.1, gamma: float = 1.0,
               n_planning_steps: int = 10, num_episodes: int = 5000, max_steps: int = 10000) -> dict:
        Q = defaultdict(float)
        model = {}
        observed_actions_per_state = defaultdict(list)
        observed_states = []

        for _ in range(num_episodes):
            env.reset()

            steps = 0
            while not env.is_game_over() and steps < max_steps:
                s = env.current_state()
                available_actions = list(env.available_actions())
                a = self._epsilon_greedy(Q, s, available_actions, epsilon)

                r = self._reward_of_step(env, a)
                s_p = env.current_state()
                s_p_is_terminal = env.is_game_over()
                available_actions_p = [] if s_p_is_terminal else list(env.available_actions())

                best_next = 0.0 if s_p_is_terminal else max(Q[(s_p, a_p)] for a_p in available_actions_p)
                Q[(s, a)] += alpha * (r + gamma * best_next - Q[(s, a)])

                if (s, a) not in model:
                    if s not in observed_actions_per_state:
                        observed_states.append(s)
                    observed_actions_per_state[s].append(a)
                model[(s, a)] = (r, s_p, available_actions_p)

                for _ in range(n_planning_steps):
                    s_sim = observed_states[np.random.randint(len(observed_states))]
                    a_sim = observed_actions_per_state[s_sim][np.random.randint(len(observed_actions_per_state[s_sim]))]
                    r_sim, s_p_sim, available_actions_p_sim = model[(s_sim, a_sim)]
                    best_next_sim = 0.0 if not available_actions_p_sim else max(Q[(s_p_sim, a_p)] for a_p in available_actions_p_sim)
                    Q[(s_sim, a_sim)] += alpha * (r_sim + gamma * best_next_sim - Q[(s_sim, a_sim)])

                steps += 1

        return Q
