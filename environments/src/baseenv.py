from abc import ABC, abstractmethod

class EnvTemplate(ABC):

    @abstractmethod
    def reset(self):
        return NotImplementedError
    
    @abstractmethod
    def step(self, action: int):
        return NotImplementedError
    
    @abstractmethod
    def current_state(self):
        return NotImplementedError
    
    @abstractmethod
    def score(self):
        return NotImplementedError
    
    @abstractmethod
    def available_actions(self):
        return NotImplementedError
    
    @abstractmethod
    def game_over(self):
        return NotImplementedError
    
    @abstractmethod
    def max_states(self):
        return NotImplementedError
    
    @abstractmethod
    def max_actions(self):
        return NotImplementedError
    
    def pretty_print(self):
        return NotImplementedError
    