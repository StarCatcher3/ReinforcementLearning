from environnements.src.secret_envs_wrapper import SecretEnv1
from algorithmes.src.monte_carlo import MonteCarlo
from algorithmes.src.temporal_difference import TemporalDifference
from algorithmes.src.planning import Planning
from experimentations.utils import evaluate_greedy_from_Q, evaluate_deterministic_policy, \
    evaluate_stochastic_policy, print_comparison_table

GAMMA = 0.99
NUM_EPISODES = 5000
EVAL_EPISODES = 500

results = {}

mc = MonteCarlo()

_, pi_mces = mc.monte_carlo_es(lambda: SecretEnv1.from_random_state(), gamma=GAMMA, num_episodes=NUM_EPISODES)
results["Monte Carlo ES"] = evaluate_deterministic_policy(SecretEnv1(), pi_mces, EVAL_EPISODES)

_, pi_onmc = mc.on_policy_first_visit_mc_control(SecretEnv1(), gamma=GAMMA, num_episodes=NUM_EPISODES)
results["On-policy first-visit MC control"] = evaluate_stochastic_policy(SecretEnv1(), pi_onmc, EVAL_EPISODES)

_, pi_offmc = mc.off_policy_mc_control(SecretEnv1(), gamma=GAMMA, num_episodes=NUM_EPISODES)
results["Off-policy MC control"] = evaluate_deterministic_policy(SecretEnv1(), pi_offmc, EVAL_EPISODES)

td = TemporalDifference()

Q_sarsa = td.sarsa(SecretEnv1(), gamma=GAMMA, num_episodes=NUM_EPISODES)
results["Sarsa"] = evaluate_greedy_from_Q(SecretEnv1(), Q_sarsa, EVAL_EPISODES)

Q_qlearning = td.q_learning(SecretEnv1(), gamma=GAMMA, num_episodes=NUM_EPISODES)
results["Q-Learning"] = evaluate_greedy_from_Q(SecretEnv1(), Q_qlearning, EVAL_EPISODES)

planning = Planning()

Q_dynaq = planning.dyna_q(SecretEnv1(), gamma=GAMMA, n_planning_steps=10, num_episodes=NUM_EPISODES)
results["Dyna-Q"] = evaluate_greedy_from_Q(SecretEnv1(), Q_dynaq, EVAL_EPISODES)

print("SecretEnv1 -- environnement mystère, pas de score optimal théorique connu")
print_comparison_table(results)
