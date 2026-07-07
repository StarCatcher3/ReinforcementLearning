from .baseenv import EnvTemplate
import numpy as np

class RockPaperScissors(EnvTemplate):

    def __init__(self, round_count: int = 2):
        self.current_score = 0
        self.round_count = round_count
        self.state = []

    # MDP related Methods
    def num_states(self) -> int:
        return 3 ** self.round_count

    def num_actions(self) -> int:
        return 3

    def num_rewards(self) -> int:
        return 3

    def reward(self, i: int) -> float:
        return i - 1
    
    def p(self, s: int, a: int, s_p: int, r_index: int) -> float:
        return NotImplementedError

    # Monte Carlo and TD Methods related functions:
    def state_id(self) -> int:
        return self.state

    def reset(self):
        self.current_score = 0
        self.state = []
    
    def display(self):
        for (i, moves) in enumerate(self.state):
            print(f"Round {i + 1}:")
            print(f"    Your Move: {moves[0]}")
            print(f"    Their Move: {moves[1]}")
            print(f"    Cumulative Score: {self.current_score}")
    
    def is_forbidden(self, action: int) -> int:
        return NotImplementedError
    
    def is_game_over(self) -> bool:
        return self.round() == self.round_count
    
    def available_actions(self) -> np.ndarray:
        return [0, 1, 2]
    
    def action_labels(self):
        return {0: "Rock", 1: "Paper", 2: "Scissors"}
    
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
    
    def score(self):
        return self.current_score
    
    @staticmethod
    def from_random_state() -> 'EnvTemplate':
        return NotImplementedError
    
    # Environment specific methods
    def round(self):
        return len(self.state)