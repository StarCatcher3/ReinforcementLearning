from environnements.src.rockpaperscissors import RockPaperScissors
from algorithmes.src.dynamic_programming import DynamicProgramming
from algorithmes.src.monte_carlo import MonteCarlo
from algorithmes.src.temporal_difference import TemporalDifference
from algorithmes.src.planning import Planning
from experimentations.utils import evaluate_greedy_from_Q, evaluate_deterministic_policy, \
    evaluate_stochastic_policy, print_comparison_table

ROUND_COUNT = 2
GAMMA = 0.99
NUM_EPISODES = 5000
EVAL_EPISODES = 1000

results = {}

dp = DynamicProgramming()

_, pi_pit = dp.policy_iteration(RockPaperScissors(ROUND_COUNT), gamma=GAMMA)
results["Policy Iteration"] = evaluate_deterministic_policy(RockPaperScissors(ROUND_COUNT), pi_pit, EVAL_EPISODES)

_, pi_vit = dp.value_iteration(RockPaperScissors(ROUND_COUNT), gamma=GAMMA)
results["Value Iteration"] = evaluate_deterministic_policy(RockPaperScissors(ROUND_COUNT), pi_vit, EVAL_EPISODES)

mc = MonteCarlo()

_, pi_onmc = mc.on_policy_first_visit_mc_control(RockPaperScissors(ROUND_COUNT), gamma=GAMMA,
                                                  num_episodes=NUM_EPISODES)
results["On-policy first-visit MC control"] = evaluate_stochastic_policy(RockPaperScissors(ROUND_COUNT), pi_onmc,
                                                                          EVAL_EPISODES)

_, pi_offmc = mc.off_policy_mc_control(RockPaperScissors(ROUND_COUNT), gamma=GAMMA, num_episodes=NUM_EPISODES)
results["Off-policy MC control"] = evaluate_deterministic_policy(RockPaperScissors(ROUND_COUNT), pi_offmc,
                                                                  EVAL_EPISODES)

td = TemporalDifference()

Q_sarsa = td.sarsa(RockPaperScissors(ROUND_COUNT), gamma=GAMMA, num_episodes=NUM_EPISODES)
results["Sarsa"] = evaluate_greedy_from_Q(RockPaperScissors(ROUND_COUNT), Q_sarsa, EVAL_EPISODES)

Q_qlearning = td.q_learning(RockPaperScissors(ROUND_COUNT), gamma=GAMMA, num_episodes=NUM_EPISODES)
results["Q-Learning"] = evaluate_greedy_from_Q(RockPaperScissors(ROUND_COUNT), Q_qlearning, EVAL_EPISODES)

planning = Planning()

Q_dynaq = planning.dyna_q(RockPaperScissors(ROUND_COUNT), gamma=GAMMA, n_planning_steps=10, num_episodes=NUM_EPISODES)
results["Dyna-Q"] = evaluate_greedy_from_Q(RockPaperScissors(ROUND_COUNT), Q_dynaq, EVAL_EPISODES)

print(f"Two-round Rock Paper Scissors -- score moyen optimal théorique = 1.0 "
      "(round 1 aléatoire donc 0 en moyenne, round 2 gagné à coup sûr en jouant "
      "le coup qui bat son propre coup du round 1)")
print_comparison_table(results)
