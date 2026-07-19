from environnements.src.baseenv import EnvTemplate
from collections import defaultdict
from typing import Callable
import numpy as np


class MonteCarlo():

    @staticmethod
    def _reward_of_step(env: EnvTemplate, action: int) -> float:
        score_before = env.score()
        env.step(action)
        return env.score() - score_before

    @staticmethod
    def _argmax_available(Q: dict, s: int, available_actions) -> int:
        best_action = None
        best_value = 0.0
        for a in available_actions:
            total = Q[(s, a)]
            if best_action is None or total >= best_value:
                best_action = a
                best_value = total
        return best_action

    def monte_carlo_es(self, env_factory: Callable[[], EnvTemplate], gamma: float = 1.0,
                        num_episodes: int = 5000, max_steps: int = 10000) -> tuple[dict, dict]:
        Q = defaultdict(float)
        Q_counts = defaultdict(int)
        pi = {}

        for _ in range(num_episodes):
            env = env_factory()

            trajectory = []
            s0 = env.current_state()
            available_actions_0 = list(env.available_actions())
            a0 = np.random.choice(available_actions_0)
            r0 = self._reward_of_step(env, a0)
            trajectory.append((s0, a0, r0, available_actions_0))

            steps = 0
            while not env.is_game_over() and steps < max_steps:
                s = env.current_state()
                available_actions = list(env.available_actions())
                a = pi.get(s, np.random.choice(available_actions))
                if a not in available_actions:
                    a = np.random.choice(available_actions)
                r = self._reward_of_step(env, a)
                trajectory.append((s, a, r, available_actions))
                steps += 1

            first_visit_index = {}
            for t, (s, a, _, _) in enumerate(trajectory):
                if (s, a) not in first_visit_index:
                    first_visit_index[(s, a)] = t

            G = 0.0
            for t in reversed(range(len(trajectory))):
                s, a, r, available_actions = trajectory[t]
                G = gamma * G + r
                if first_visit_index[(s, a)] == t:
                    Q[(s, a)] = (Q[(s, a)] * Q_counts[(s, a)] + G) / (Q_counts[(s, a)] + 1)
                    Q_counts[(s, a)] += 1
                    pi[s] = self._argmax_available(Q, s, available_actions)

        return Q, pi

    def on_policy_first_visit_mc_control(self, env: EnvTemplate, epsilon: float = 0.1, gamma: float = 1.0,
                                          num_episodes: int = 5000, max_steps: int = 10000) -> tuple[dict, dict]:
        Q = defaultdict(float)
        Q_counts = defaultdict(int)
        pi = {}

        def action_probabilities(s, available_actions):
            if s not in pi:
                pi[s] = {a: 1.0 / len(available_actions) for a in available_actions}
            return pi[s]

        def sample_action(s, available_actions):
            probs = action_probabilities(s, available_actions)
            actions = list(probs.keys())
            weights = [probs[a] for a in actions]
            return np.random.choice(actions, p=weights)

        for _ in range(num_episodes):
            env.reset()
            trajectory = []
            steps = 0
            while not env.is_game_over() and steps < max_steps:
                s = env.current_state()
                available_actions = list(env.available_actions())
                a = sample_action(s, available_actions)
                r = self._reward_of_step(env, a)
                trajectory.append((s, a, r, available_actions))
                steps += 1

            first_visit_index = {}
            for t, (s, a, _, _) in enumerate(trajectory):
                if (s, a) not in first_visit_index:
                    first_visit_index[(s, a)] = t

            G = 0.0
            for t in reversed(range(len(trajectory))):
                s, a, r, available_actions = trajectory[t]
                G = gamma * G + r
                if first_visit_index[(s, a)] == t:
                    Q[(s, a)] = (Q[(s, a)] * Q_counts[(s, a)] + G) / (Q_counts[(s, a)] + 1)
                    Q_counts[(s, a)] += 1
                    best_action = self._argmax_available(Q, s, available_actions)
                    for a_prime in available_actions:
                        if a_prime == best_action:
                            pi[s][a_prime] = 1 - epsilon + epsilon / len(available_actions)
                        else:
                            pi[s][a_prime] = epsilon / len(available_actions)

        return Q, pi

    def off_policy_mc_control(self, env: EnvTemplate, gamma: float = 1.0,
                               num_episodes: int = 5000, max_steps: int = 10000) -> tuple[dict, dict]:
        Q = defaultdict(float)
        C = defaultdict(float)
        pi = {}

        for _ in range(num_episodes):
            env.reset()

            trajectory = []
            steps = 0
            while not env.is_game_over() and steps < max_steps:
                s = env.current_state()
                available_actions = list(env.available_actions())
                b_probabilities = 1.0 / len(available_actions)
                a = np.random.choice(available_actions)
                r = self._reward_of_step(env, a)
                trajectory.append((s, a, r, available_actions, b_probabilities))
                steps += 1

            G = 0.0
            W = 1.0
            for t in reversed(range(len(trajectory))):
                s, a, r, available_actions, b_prob = trajectory[t]
                G = gamma * G + r
                C[(s, a)] += W
                Q[(s, a)] += (W / C[(s, a)]) * (G - Q[(s, a)])
                pi[s] = self._argmax_available(Q, s, available_actions)
                if a != pi[s]:
                    break
                W *= 1.0 / b_prob

        return Q, pi
