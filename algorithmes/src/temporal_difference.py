from environnements.src.baseenv import EnvTemplate
from collections import defaultdict
import numpy as np


class TemporalDifference():
    """Model-free Temporal Difference control methods (Sutton & Barto, chap. 6).

    As for Monte Carlo methods, these algorithms never call env.p(...): the
    immediate reward R is read as the variation of env.score() before/after
    each env.step(...) call.
    """

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

    def sarsa(self, env: EnvTemplate, alpha: float = 0.1, epsilon: float = 0.1, gamma: float = 1.0,
              num_episodes: int = 5000, max_steps: int = 10000) -> dict:
        """Sarsa (on-policy TD control), for estimating Q ~= q_* (Sutton & Barto, chap. 6)."""
        Q = defaultdict(float)

        for _ in range(num_episodes):
            env.reset()
            s = env.current_state()
            available_actions = list(env.available_actions())
            a = self._epsilon_greedy(Q, s, available_actions, epsilon)

            steps = 0
            while not env.is_game_over() and steps < max_steps:
                r = self._reward_of_step(env, a)
                s_p = env.current_state()

                if env.is_game_over():
                    Q[(s, a)] += alpha * (r + gamma * 0.0 - Q[(s, a)])
                    break

                available_actions_p = list(env.available_actions())
                a_p = self._epsilon_greedy(Q, s_p, available_actions_p, epsilon)

                Q[(s, a)] += alpha * (r + gamma * Q[(s_p, a_p)] - Q[(s, a)])

                s, a = s_p, a_p
                steps += 1

        return Q

    def q_learning(self, env: EnvTemplate, alpha: float = 0.1, epsilon: float = 0.1, gamma: float = 1.0,
                   num_episodes: int = 5000, max_steps: int = 10000) -> dict:
        """Q-learning (off-policy TD control), for estimating pi ~= pi_* (Sutton & Barto, chap. 6)."""
        Q = defaultdict(float)

        for _ in range(num_episodes):
            env.reset()
            s = env.current_state()

            steps = 0
            while not env.is_game_over() and steps < max_steps:
                available_actions = list(env.available_actions())
                a = self._epsilon_greedy(Q, s, available_actions, epsilon)
                r = self._reward_of_step(env, a)
                s_p = env.current_state()

                if env.is_game_over():
                    Q[(s, a)] += alpha * (r + gamma * 0.0 - Q[(s, a)])
                    break

                best_next = max(Q[(s_p, a_p)] for a_p in env.available_actions())
                Q[(s, a)] += alpha * (r + gamma * best_next - Q[(s, a)])

                s = s_p
                steps += 1

        return Q
