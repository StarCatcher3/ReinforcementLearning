from environnements.src.lineworld import LineWorld
from environnements.src.gridworld import GridWorld
from environnements.src.rockpaperscissors import RockPaperScissors
from environnements.src.secret_envs_wrapper import SecretEnv0
from environnements.src.secret_envs_wrapper import SecretEnv1
from environnements.src.secret_envs_wrapper import SecretEnv2
from environnements.src.secret_envs_wrapper import SecretEnv3

env = RockPaperScissors(4)

print("Starting game:")
env.pretty_print()

while not env.is_game_over():
    action = input(f"Choose an action from the following choices: {env.available_actions()}\n")
    try:
        action_id = int(action)
    except:
        print("Invalid Input")
        continue
    if action_id in env.available_actions():
        env.step(action_id)
        env.pretty_print()
    if action_id == -1:
        env.reset()
        print("Resetting game")
        env.pretty_print()

print(f"Game over with final score of {env.score()}")
