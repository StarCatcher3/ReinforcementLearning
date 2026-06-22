from .baseenv import EnvTemplate
import numpy as np

class RockPaperScissors(EnvTemplate):

    def __init__(self, round_count: int = 2):
        self.current_score = 0
        self.round_count = round_count
        self.state = []

    def reset(self):
        self.current_score = 0
        self.state = []

    def step(self, action: int):
        if action not in self.available_actions():
            raise Exception("Action not allowed in current state")
        
        if self.game_over():
            raise Exception("Game is already over")
        
        opponent_action = np.random.randint(0, 2) if self.round() == 0 else self.state[-1][0]
        self.state.append((action, opponent_action))
        
        if (action == 0 and opponent_action == 1) or (action == 1 and opponent_action == 2) or (action == 2 and opponent_action == 0):
            self.current_score -= 1

        if (action == 0 and opponent_action == 2) or (action == 1 and opponent_action == 0) or (action == 2 and opponent_action == 1):
            self.current_score += 1

    def current_state(self):
        return self.state

    def score(self):
        return self.current_score

    def available_actions(self):
        return [0, 1, 2]
    
    def action_labels(self):
        return {0: "Rock", 1: "Paper", 2: "Scissors"}
    
    def game_over(self):
        return self.round() == self.round_count
    
    def round(self):
        return len(self.state)
    
    def max_states(self):
        return 3 ** self.round_count
    
    def max_actions(self):
        return 3

    def pretty_print(self):
        for (i, moves) in enumerate(self.state):
            print(f"Round {i + 1}:")
            print(f"    Your Move: {moves[0]}")
            print(f"    Their Move: {moves[1]}")
            print(f"    Cumulative Score: {self.current_score}")