from abc import ABC
import numpy as np

class EnvTemplate(ABC):

    # MDP related Methods
    def num_states(self) -> int:
        return NotImplementedError

    def num_actions(self) -> int:
        return NotImplementedError

    def num_rewards(self) -> int:
        return NotImplementedError

    def reward(self, i: int) -> float:
        return NotImplementedError
    
    def p(self, s: int, a: int, s_p: int, r_index: int) -> float:
        return NotImplementedError

    # Monte Carlo and TD Methods related functions:
    def state_id(self) -> int:
        return NotImplementedError

    def reset(self):
        return NotImplementedError
    
    def display(self):
        return NotImplementedError
    
    def is_forbidden(self, action: int) -> int:
        return NotImplementedError
    
    def is_game_over(self) -> bool:
        return NotImplementedError
    
    def available_actions(self) -> np.ndarray:
        return NotImplementedError
    
    def step(self, action: int):
        return NotImplementedError
    
    def score(self):
        return NotImplementedError
    
    @staticmethod
    def from_random_state() -> 'EnvTemplate':
        return NotImplementedError
    