from environnements.src.baseenv import EnvTemplate
import numpy as np


def evaluate_greedy_from_Q(env: EnvTemplate, Q: dict, num_episodes: int = 200, max_steps: int = 1000) -> float:
    total = 0.0
    for _ in range(num_episodes):
        env.reset()
        steps = 0
        while not env.is_game_over() and steps < max_steps:
            s = env.current_state()
            available_actions = list(env.available_actions())
            best_action = None
            best_value = 0.0
            for a in available_actions:
                action_value = Q[(s, a)]
                if best_action is None or action_value >= best_value:
                    best_action = a
                    best_value = action_value
            env.step(best_action)
            steps += 1
        total += env.score()
    return total / num_episodes


def evaluate_deterministic_policy(env: EnvTemplate, pi, num_episodes: int = 200, max_steps: int = 1000) -> float:
    is_dict = isinstance(pi, dict)
    total = 0.0
    for _ in range(num_episodes):
        env.reset()
        steps = 0
        while not env.is_game_over() and steps < max_steps:
            s = env.current_state()
            available_actions = list(env.available_actions())
            has_action = (s in pi) if is_dict else (0 <= s < len(pi))
            if has_action:
                action = pi[s]
                if action not in available_actions:
                    action = np.random.choice(available_actions)
            else:
                action = np.random.choice(available_actions)
            env.step(action)
            steps += 1
        total += env.score()
    return total / num_episodes


def evaluate_stochastic_policy(env: EnvTemplate, pi: dict, num_episodes: int = 200, max_steps: int = 1000) -> float:
    total = 0.0
    for _ in range(num_episodes):
        env.reset()
        steps = 0
        while not env.is_game_over() and steps < max_steps:
            s = env.current_state()
            available_actions = list(env.available_actions())
            if s in pi:
                actions = list(pi[s].keys())
                weights = [pi[s][a] for a in actions]
                action = np.random.choice(actions, p=weights)
            else:
                action = np.random.choice(available_actions)
            env.step(action)
            steps += 1
        total += env.score()
    return total / num_episodes


def print_comparison_table(results: dict):
    print("\n=== Comparaison des algorithmes (score moyen) ===")
    for name, avg_score in results.items():
        print(f"{name:35s} : {avg_score:.4f}")
