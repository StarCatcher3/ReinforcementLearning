from environnements.src.secret_envs_wrapper import SecretEnv0
from algorithmes.src.monte_carlo import MonteCarlo
from algorithmes.src.temporal_difference import TemporalDifference
from algorithmes.src.planning import Planning
from experimentations.utils import evaluate_greedy_from_Q, evaluate_deterministic_policy, \
    evaluate_stochastic_policy, print_comparison_table

# NOTE: Dynamic Programming (Policy/Value Iteration) is intentionally not run here.
# SecretEnv0 has 8192 states; a single full sweep of Iterative Policy Evaluation
# needs S*A*S*R calls to env.p(...) (here ~600 million), which is not tractable
# through the ctypes wrapper. Only model-free (MC/TD/Planning) methods are used,
# as explicitly allowed by the syllabus ("quand possible").

GAMMA = 0.99
NUM_EPISODES = 5000
EVAL_EPISODES = 500

results = {}

# Monte Carlo
mc = MonteCarlo()

_, pi_mces = mc.monte_carlo_es(lambda: SecretEnv0.from_random_state(), gamma=GAMMA, num_episodes=NUM_EPISODES)
results["Monte Carlo ES"] = evaluate_deterministic_policy(SecretEnv0(), pi_mces, EVAL_EPISODES)

_, pi_onmc = mc.on_policy_first_visit_mc_control(SecretEnv0(), gamma=GAMMA, num_episodes=NUM_EPISODES)
results["On-policy first-visit MC control"] = evaluate_stochastic_policy(SecretEnv0(), pi_onmc, EVAL_EPISODES)

_, pi_offmc = mc.off_policy_mc_control(SecretEnv0(), gamma=GAMMA, num_episodes=NUM_EPISODES)
results["Off-policy MC control"] = evaluate_deterministic_policy(SecretEnv0(), pi_offmc, EVAL_EPISODES)

# Temporal Difference
td = TemporalDifference()

Q_sarsa = td.sarsa(SecretEnv0(), gamma=GAMMA, num_episodes=NUM_EPISODES)
results["Sarsa"] = evaluate_greedy_from_Q(SecretEnv0(), Q_sarsa, EVAL_EPISODES)

Q_qlearning = td.q_learning(SecretEnv0(), gamma=GAMMA, num_episodes=NUM_EPISODES)
results["Q-Learning"] = evaluate_greedy_from_Q(SecretEnv0(), Q_qlearning, EVAL_EPISODES)

# Planning
planning = Planning()

Q_dynaq = planning.dyna_q(SecretEnv0(), gamma=GAMMA, n_planning_steps=10, num_episodes=NUM_EPISODES)
results["Dyna-Q"] = evaluate_greedy_from_Q(SecretEnv0(), Q_dynaq, EVAL_EPISODES)

print("SecretEnv0 -- environnement mystère, pas de score optimal théorique connu")
print_comparison_table(results)
