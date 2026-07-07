from secret_envs_wrapper import SecretEnv0
from secret_envs_wrapper import SecretEnv1
from secret_envs_wrapper import SecretEnv2
from secret_envs_wrapper import SecretEnv3

env = SecretEnv3()

print("Starting game:")
env.display()

while not env.is_game_over():
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