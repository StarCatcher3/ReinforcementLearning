from environnements.src.montyhall2 import MontyHallLevel2
from algorithmes.src.dynamic_programming import DynamicProgramming
from algorithmes.src.monte_carlo import MonteCarlo
from algorithmes.src.temporal_difference import TemporalDifference
from algorithmes.src.planning import Planning
from experimentations.utils import evaluate_greedy_from_Q, evaluate_deterministic_policy, \
    evaluate_stochastic_policy, print_comparison_table

GAMMA = 1.0
NUM_EPISODES = 8000
EVAL_EPISODES = 3000

results = {}

dp = DynamicProgramming()

_, pi_pit = dp.policy_iteration(MontyHallLevel2(), gamma=GAMMA)
results["Policy Iteration"] = evaluate_deterministic_policy(MontyHallLevel2(), pi_pit, EVAL_EPISODES)

_, pi_vit = dp.value_iteration(MontyHallLevel2(), gamma=GAMMA)
results["Value Iteration"] = evaluate_deterministic_policy(MontyHallLevel2(), pi_vit, EVAL_EPISODES)

mc = MonteCarlo()

_, pi_mces = mc.monte_carlo_es(lambda: MontyHallLevel2.from_random_state(), gamma=GAMMA, num_episodes=NUM_EPISODES)
results["Monte Carlo ES"] = evaluate_deterministic_policy(MontyHallLevel2(), pi_mces, EVAL_EPISODES)

_, pi_onmc = mc.on_policy_first_visit_mc_control(MontyHallLevel2(), gamma=GAMMA, num_episodes=NUM_EPISODES)
results["On-policy first-visit MC control"] = evaluate_stochastic_policy(MontyHallLevel2(), pi_onmc, EVAL_EPISODES)

_, pi_offmc = mc.off_policy_mc_control(MontyHallLevel2(), gamma=GAMMA, num_episodes=NUM_EPISODES)
results["Off-policy MC control"] = evaluate_deterministic_policy(MontyHallLevel2(), pi_offmc, EVAL_EPISODES)

td = TemporalDifference()

Q_sarsa = td.sarsa(MontyHallLevel2(), gamma=GAMMA, num_episodes=NUM_EPISODES)
results["Sarsa"] = evaluate_greedy_from_Q(MontyHallLevel2(), Q_sarsa, EVAL_EPISODES)

Q_qlearning = td.q_learning(MontyHallLevel2(), gamma=GAMMA, num_episodes=NUM_EPISODES)
results["Q-Learning"] = evaluate_greedy_from_Q(MontyHallLevel2(), Q_qlearning, EVAL_EPISODES)

planning = Planning()

Q_dynaq = planning.dyna_q(MontyHallLevel2(), gamma=GAMMA, n_planning_steps=10, num_episodes=NUM_EPISODES)
results["Dyna-Q"] = evaluate_greedy_from_Q(MontyHallLevel2(), Q_dynaq, EVAL_EPISODES)

print("Monty Hall niveau 2 -- score optimal théorique = 4/5 "
      "(garder son choix initial et ne changer qu'à la toute dernière décision)")
print_comparison_table(results)
