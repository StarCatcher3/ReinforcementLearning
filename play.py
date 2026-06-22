from environments.src.lineworld import LineWorld
from environments.src.gridworld import GridWorld
from environments.src.rockpaperscissors import RockPaperScissors

#env = LineWorld(9)
#env = GridWorld(7, 3)
env = RockPaperScissors(5)

print("Starting game:")
env.pretty_print()

while not env.game_over():
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