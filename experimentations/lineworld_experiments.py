from environnements.src.lineworld import LineWorld
from algorithmes.src.dynamic_programming import DynamicProgramming
from algorithmes.src.monte_carlo import MonteCarlo
from algorithmes.src.temporal_difference import TemporalDifference
from algorithmes.src.planning import Planning
from experimentations.utils import evaluate_greedy_from_Q, evaluate_deterministic_policy, \
    evaluate_stochastic_policy, print_comparison_table

NUM_CELLS = 7
GAMMA = 0.99
NUM_EPISODES = 3000
EVAL_EPISODES = 500

results = {}

# Dynamic Programming
dp = DynamicProgramming()

_, pi_pit = dp.policy_iteration(LineWorld(NUM_CELLS), gamma=GAMMA)
results["Policy Iteration"] = evaluate_deterministic_policy(LineWorld(NUM_CELLS), pi_pit, EVAL_EPISODES)

_, pi_vit = dp.value_iteration(LineWorld(NUM_CELLS), gamma=GAMMA)
results["Value Iteration"] = evaluate_deterministic_policy(LineWorld(NUM_CELLS), pi_vit, EVAL_EPISODES)

# Monte Carlo
mc = MonteCarlo()

_, pi_mces = mc.monte_carlo_es(lambda: LineWorld.from_random_state(NUM_CELLS), gamma=GAMMA, num_episodes=NUM_EPISODES)
results["Monte Carlo ES"] = evaluate_deterministic_policy(LineWorld(NUM_CELLS), pi_mces, EVAL_EPISODES)

_, pi_onmc = mc.on_policy_first_visit_mc_control(LineWorld(NUM_CELLS), gamma=GAMMA, num_episodes=NUM_EPISODES)
results["On-policy first-visit MC control"] = evaluate_stochastic_policy(LineWorld(NUM_CELLS), pi_onmc, EVAL_EPISODES)

_, pi_offmc = mc.off_policy_mc_control(LineWorld(NUM_CELLS), gamma=GAMMA, num_episodes=NUM_EPISODES)
results["Off-policy MC control"] = evaluate_deterministic_policy(LineWorld(NUM_CELLS), pi_offmc, EVAL_EPISODES)

# Temporal Difference
td = TemporalDifference()

Q_sarsa = td.sarsa(LineWorld(NUM_CELLS), gamma=GAMMA, num_episodes=NUM_EPISODES)
results["Sarsa"] = evaluate_greedy_from_Q(LineWorld(NUM_CELLS), Q_sarsa, EVAL_EPISODES)

Q_qlearning = td.q_learning(LineWorld(NUM_CELLS), gamma=GAMMA, num_episodes=NUM_EPISODES)
results["Q-Learning"] = evaluate_greedy_from_Q(LineWorld(NUM_CELLS), Q_qlearning, EVAL_EPISODES)

# Planning
planning = Planning()

Q_dynaq = planning.dyna_q(LineWorld(NUM_CELLS), gamma=GAMMA, n_planning_steps=10, num_episodes=NUM_EPISODES)
results["Dyna-Q"] = evaluate_greedy_from_Q(LineWorld(NUM_CELLS), Q_dynaq, EVAL_EPISODES)

print(f"LineWorld({NUM_CELLS}) -- score optimal théorique = 1.0 (toujours aller à droite)")
print_comparison_table(results)
