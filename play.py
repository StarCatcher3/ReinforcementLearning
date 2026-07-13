from environments.src.lineworld import LineWorld
from environments.src.gridworld import GridWorld
from environments.src.rockpaperscissors import RockPaperScissors
from environments.src.secret_envs_wrapper import SecretEnv0
from environments.src.secret_envs_wrapper import SecretEnv1
from environments.src.secret_envs_wrapper import SecretEnv2
from environments.src.secret_envs_wrapper import SecretEnv3

#env = LineWorld(9)
#env = GridWorld(7, 3)
#env = RockPaperScissors(4)
env = SecretEnv0()
#env = GridWorld.from_random_state(3, 3)

print("Starting game:")
env.display()

while not env.is_game_over():
    print(f"New game state: {env.state_id()}")
    action = input(f"Choose an action from the following choices: {env.available_actions()}\n")
    try:
        action_id = int(action)
    except:
        print("Invalid Input")
        continue
    if action_id in env.available_actions():
        env.step(action_id)
        env.display()
    if action_id == -1:
        env.reset()
        print("Resetting game")
        env.display()

print(f"Game over with final score of {env.score()}")