from .baseenv import EnvTemplate
import numpy as np

class RockPaperScissors(EnvTemplate):

    def __init__(self, round_count: int = 2):
        self.current_score = 0
        self.current_round = 0
        self.round_count = round_count
        self.history = ()

    # MDP related Methods
    def num_states(self) -> int:
        return 2 + (self.round_count - 1) * 3

    def num_actions(self) -> int:
        return 3

    def num_rewards(self) -> int:
        return 3

    def reward(self, i: int) -> float:
        return [0, 1, -1][i]
    
    def p(self, s: int, a: int, s_p: int, r_index: int) -> float:
        if s == self.num_states() - 1:
            return 0.0
        if s == 0 and s_p == a + 1:
            return 1 / 3
        last_move = (s + 2) % 3
        if s != 0 and (a - last_move) % 3 == r_index:
            round = (s + 2) // 3
            if round == self.round_count - 1:
                if (s_p == self.num_states() - 1):
                    return 1.0
            elif s_p == round * 3 - 2 + a:
                return 1.0
                
        return 0.0

    # Monte Carlo and TD Methods related functions:
    def state_id(self) -> int:
        if self.current_round == 0: return 0
        if self.is_game_over(): return self.num_states - 1
        return (self.current_round - 1) * 3 + self.history[0] + 1
    
    def state_desc(self, i: int):
        if self.current_round == 0: return "First Round"
        if self.is_game_over(): return "End of Game"
        return ["Last Played Scissors", "Last Played Rock", "Last Played Paper"][i % 3]

    def reset(self):
        self.current_score = 0
        self.current_round = 0
        self.history = ()
    
    def display(self):
        if self.current_round > 0:
            print(f"Round {self.current_round}:")
            print(f"    Your Move: {self.history[0]}")
            print(f"    Their Move: {self.history[1]}")
            print(f"    Cumulative Score: {self.current_score}")
    
    def is_forbidden(self, action: int) -> int:
        return False
    
    def is_game_over(self) -> bool:
        return self.current_round == self.round_count
    
    def available_actions(self) -> np.ndarray:
        return [0, 1, 2]
    
    def action_desc(self):
        return {0: "Rock", 1: "Paper", 2: "Scissors"}
    
    def step(self, action: int):
        if action not in self.available_actions():
            raise Exception("Action not allowed in current state")
        
        if self.is_game_over():
            raise Exception("Game is already over")
        
        opponent_action = np.random.randint(0, 2) if self.current_round == 0 else self.history[0]
        self.history = (action, opponent_action)

        # (My action - Opponent action) Mod 3 == Reward_Id
        self.current_score += self.reward((action - opponent_action) % 3)
        
        self.current_round += 1
    
    def score(self):
        return self.current_score
    
    @staticmethod
    def from_random_state() -> 'EnvTemplate':
        return NotImplementedError