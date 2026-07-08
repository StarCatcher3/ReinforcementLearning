from .baseenv import EnvTemplate
import numpy as np

class LineWorld(EnvTemplate):


    def __init__(self, num_cells: int = 5, instance=None):
        if instance is None:
            self.num_cells = num_cells
            self.pos = num_cells // 2
        self.instance = instance

    # MDP related Methods
    def num_states(self) -> int:
        return self.num_cells

    def num_actions(self) -> int:
        return 2

    def num_rewards(self) -> int:
        return 3

    def reward(self, i: int) -> float:
        return i - 1
    
    def p(self, s: int, a: int, s_p: int, r_index: int) -> float:
        # Return 0 if in an end state
        if s == 0 or s == self.num_cells - 1:
            return 0.0
        
        if s + (a * 2 - 1) == s_p:
            r = self.reward(r_index)
            if s_p == self.num_cells - 1 and r == 1:
                return 1.0
            elif s_p == 0 and r == -1:
                return 1.0
            elif r == 0:
                return 1.0
        return 0.0

    # Monte Carlo and TD Methods related functions:
    def state_id(self) -> int:
        return self.pos

    def reset(self):
        self.pos = self.num_cells // 2
    
    def display(self):
        for i in range(self.num_cells):
            if i == self.pos:
                print("|X", end="")
            else:
                print("|_", end="")
        print("|")
    
    def is_forbidden(self, action: int) -> int:
        return False
    
    def is_game_over(self) -> bool:
        return self.pos == 0 or self.pos == self.num_cells - 1
    
    def available_actions(self) -> np.ndarray:
        return [0, 1]
    
    def action_desc(self):
        return {0: "Left", 1: "Right"}
    
    def step(self, action: int):
        if action not in self.available_actions():
            raise Exception("Action not allowed in current state")
        
        if self.is_game_over():
            raise Exception("Game is already over")
        
        self.pos += action * 2 - 1
    
    def score(self):
        if self.pos == 0:
            return -1
        elif self.pos == self.num_cells - 1:
            return 1
        else:
            return 0
        
    @staticmethod
    def from_random_state(num_cells: int = 5) -> 'LineWorld':
        instance = LineWorld(num_cells)
        instance.pos = np.random.choice(range(1, num_cells - 1))
        return instance