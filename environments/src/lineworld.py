from .baseenv import EnvTemplate

class LineWorld(EnvTemplate):

    def __init__(self, num_cells: int = 5):
        self.num_cells = num_cells
        self.pos = num_cells // 2

    def reset(self):
        self.pos = self.num_cells // 2

    def step(self, action: int):
        if action not in self.available_actions():
            raise Exception("Action not allowed in current state")
        
        if self.game_over():
            raise Exception("Game is already over")
        
        self.pos += action * 2 - 1

    def current_state(self):
        return self.pos

    def score(self):
        if self.pos == 0:
            return -1
        elif self.pos == self.num_cells - 1:
            return 1
        else:
            return 0

    def available_actions(self):
        return [0, 1]
    
    def action_labels(self):
        return {0: "Left", 1: "Right"}
    
    def game_over(self):
        return self.pos == 0 or self.pos == self.num_cells - 1
    
    def max_states(self):
        return self.num_cells
    
    def max_actions(self):
        return 2

    def pretty_print(self):
        for i in range(self.num_cells):
            if i == self.pos:
                print("|X", end="")
            else:
                print("|_", end="")
        print("|")