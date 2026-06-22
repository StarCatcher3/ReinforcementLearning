from .baseenv import EnvTemplate

class GridWorld(EnvTemplate):

    def __init__(self, grid_width: int = 5, grid_height: int = 5):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.pos = (0, 0)

    def reset(self):
        self.pos = (0, 0)

    def step(self, action: int):
        if action not in self.available_actions():
            raise Exception("Action not allowed in current state")
        
        if self.game_over():
            raise Exception("Game is already over")
        
        if (action == 0 and self.pos[0] > 0):
            self.pos = (self.pos[0] - 1, self.pos[1])
        if (action == 1 and self.pos[1] > 0):
            self.pos = (self.pos[0], self.pos[1] - 1)
        if (action == 2 and self.pos[0] < self.grid_width - 1):
            self.pos = (self.pos[0] + 1, self.pos[1])
        if (action == 3 and self.pos[1] < self.grid_height - 1):
            self.pos = (self.pos[0], self.pos[1] + 1)

    def current_state(self):
        return self.pos

    def score(self):
        if self.pos == (self.grid_width - 1, 0):
            return -3
        elif self.pos == (self.grid_width - 1, self.grid_height - 1):
            return 1
        else:
            return 0

    def available_actions(self):
        return [0, 1, 2, 3]
    
    def action_labels(self):
        return {0: "Left", 1: "Up", 2: "Right", 3: "Down"}
    
    def game_over(self):
        return self.pos == (self.grid_width - 1, 0) or self.pos == (self.grid_width - 1, self.grid_height - 1)
    
    def max_states(self):
        return self.grid_width * self.grid_height
    
    def max_actions(self):
        return 4

    def pretty_print(self):
        for h in range(self.grid_height):
            for w in range(self.grid_width):
                if self.pos == (w, h):
                    print("|X", end="")
                else:
                    print("|_", end="")
            print("|")