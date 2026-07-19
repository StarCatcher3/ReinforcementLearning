from environnements.src.gridworld import GridWorld
from algorithmes.src.dynamic_programming import DynamicProgramming
from algorithmes.src.monte_carlo import MonteCarlo
from algorithmes.src.temporal_difference import TemporalDifference
from algorithmes.src.planning import Planning
from experimentations.utils import evaluate_greedy_from_Q, evaluate_deterministic_policy, \
    evaluate_stochastic_policy, print_comparison_table

GRID_WIDTH = 5
GRID_HEIGHT = 5
GAMMA = 0.99
NUM_EPISODES = 5000
EVAL_EPISODES = 500

results = {}

# Dynamic Programming
dp = DynamicProgramming()

_, pi_pit = dp.policy_iteration(GridWorld(GRID_WIDTH, GRID_HEIGHT), gamma=GAMMA)
results["Policy Iteration"] = evaluate_deterministic_policy(GridWorld(GRID_WIDTH, GRID_HEIGHT), pi_pit, EVAL_EPISODES)

_, pi_vit = dp.value_iteration(GridWorld(GRID_WIDTH, GRID_HEIGHT), gamma=GAMMA)
results["Value Iteration"] = evaluate_deterministic_policy(GridWorld(GRID_WIDTH, GRID_HEIGHT), pi_vit, EVAL_EPISODES)

# Monte Carlo
mc = MonteCarlo()

_, pi_mces = mc.monte_carlo_es(lambda: GridWorld.from_random_state(GRID_WIDTH, GRID_HEIGHT), gamma=GAMMA,
                                num_episodes=NUM_EPISODES)
results["Monte Carlo ES"] = evaluate_deterministic_policy(GridWorld(GRID_WIDTH, GRID_HEIGHT), pi_mces, EVAL_EPISODES)

_, pi_onmc = mc.on_policy_first_visit_mc_control(GridWorld(GRID_WIDTH, GRID_HEIGHT), gamma=GAMMA,
                                                  num_episodes=NUM_EPISODES)
results["On-policy first-visit MC control"] = evaluate_stochastic_policy(GridWorld(GRID_WIDTH, GRID_HEIGHT),
                                                                          pi_onmc, EVAL_EPISODES)

_, pi_offmc = mc.off_policy_mc_control(GridWorld(GRID_WIDTH, GRID_HEIGHT), gamma=GAMMA, num_episodes=NUM_EPISODES)
results["Off-policy MC control"] = evaluate_deterministic_policy(GridWorld(GRID_WIDTH, GRID_HEIGHT), pi_offmc,
                                                                  EVAL_EPISODES)

# Temporal Difference
td = TemporalDifference()

Q_sarsa = td.sarsa(GridWorld(GRID_WIDTH, GRID_HEIGHT), gamma=GAMMA, num_episodes=NUM_EPISODES)
results["Sarsa"] = evaluate_greedy_from_Q(GridWorld(GRID_WIDTH, GRID_HEIGHT), Q_sarsa, EVAL_EPISODES)

Q_qlearning = td.q_learning(GridWorld(GRID_WIDTH, GRID_HEIGHT), gamma=GAMMA, num_episodes=NUM_EPISODES)
results["Q-Learning"] = evaluate_greedy_from_Q(GridWorld(GRID_WIDTH, GRID_HEIGHT), Q_qlearning, EVAL_EPISODES)

# Planning
planning = Planning()

Q_dynaq = planning.dyna_q(GridWorld(GRID_WIDTH, GRID_HEIGHT), gamma=GAMMA, n_planning_steps=20,
                           num_episodes=NUM_EPISODES)
results["Dyna-Q"] = evaluate_greedy_from_Q(GridWorld(GRID_WIDTH, GRID_HEIGHT), Q_dynaq, EVAL_EPISODES)

print(f"GridWorld({GRID_WIDTH}x{GRID_HEIGHT}) -- score optimal théorique = 1.0 (atteindre le coin (width-1, height-1))")
print_comparison_table(results)
