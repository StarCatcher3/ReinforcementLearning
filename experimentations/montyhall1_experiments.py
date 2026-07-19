from environnements.src.montyhall1 import MontyHallLevel1
from algorithmes.src.dynamic_programming import DynamicProgramming
from algorithmes.src.monte_carlo import MonteCarlo
from algorithmes.src.temporal_difference import TemporalDifference
from algorithmes.src.planning import Planning
from experimentations.utils import evaluate_greedy_from_Q, evaluate_deterministic_policy, \
    evaluate_stochastic_policy, print_comparison_table

GAMMA = 1.0
NUM_EPISODES = 3000
EVAL_EPISODES = 2000

results = {}

# Dynamic Programming
dp = DynamicProgramming()

_, pi_pit = dp.policy_iteration(MontyHallLevel1(), gamma=GAMMA)
results["Policy Iteration"] = evaluate_deterministic_policy(MontyHallLevel1(), pi_pit, EVAL_EPISODES)

_, pi_vit = dp.value_iteration(MontyHallLevel1(), gamma=GAMMA)
results["Value Iteration"] = evaluate_deterministic_policy(MontyHallLevel1(), pi_vit, EVAL_EPISODES)

# Monte Carlo
mc = MonteCarlo()

_, pi_mces = mc.monte_carlo_es(lambda: MontyHallLevel1.from_random_state(), gamma=GAMMA, num_episodes=NUM_EPISODES)
results["Monte Carlo ES"] = evaluate_deterministic_policy(MontyHallLevel1(), pi_mces, EVAL_EPISODES)

_, pi_onmc = mc.on_policy_first_visit_mc_control(MontyHallLevel1(), gamma=GAMMA, num_episodes=NUM_EPISODES)
results["On-policy first-visit MC control"] = evaluate_stochastic_policy(MontyHallLevel1(), pi_onmc, EVAL_EPISODES)

_, pi_offmc = mc.off_policy_mc_control(MontyHallLevel1(), gamma=GAMMA, num_episodes=NUM_EPISODES)
results["Off-policy MC control"] = evaluate_deterministic_policy(MontyHallLevel1(), pi_offmc, EVAL_EPISODES)

# Temporal Difference
td = TemporalDifference()

Q_sarsa = td.sarsa(MontyHallLevel1(), gamma=GAMMA, num_episodes=NUM_EPISODES)
results["Sarsa"] = evaluate_greedy_from_Q(MontyHallLevel1(), Q_sarsa, EVAL_EPISODES)

Q_qlearning = td.q_learning(MontyHallLevel1(), gamma=GAMMA, num_episodes=NUM_EPISODES)
results["Q-Learning"] = evaluate_greedy_from_Q(MontyHallLevel1(), Q_qlearning, EVAL_EPISODES)

# Planning
planning = Planning()

Q_dynaq = planning.dyna_q(MontyHallLevel1(), gamma=GAMMA, n_planning_steps=10, num_episodes=NUM_EPISODES)
results["Dyna-Q"] = evaluate_greedy_from_Q(MontyHallLevel1(), Q_dynaq, EVAL_EPISODES)

print("Monty Hall niveau 1 -- score optimal théorique = 2/3 (toujours changer de porte)")
print_comparison_table(results)
