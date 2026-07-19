from .baseenv import EnvTemplate
import numpy as np

class GridWorld(EnvTemplate):

    def __init__(self, grid_width: int = 5, grid_height: int = 5):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.pos = (0, 0)

    def maximum_states_count(self) -> int:
        return self.grid_width * self.grid_height

    def maximum_actions_count(self) -> int:
        return 4

    def num_rewards(self) -> int:
        return 3

    def reward(self, i: int) -> float:
        R = [-3, 0, 1]
        return R[i]
    
    def p(self, s: int, a: int, s_p: int, r_index: int) -> float:
        y = s // self.grid_width
        x = s - y * self.grid_width

        x_last = self.grid_width - 1
        y_last = self.grid_height - 1

        if ((x, y) == (x_last, 0)
            or (x, y) == (x_last, y_last)):
            return 0.0

        y_p = s_p // self.grid_width
        x_p = s_p - y_p * self.grid_width

        r = self.reward(r_index)
        if ((a == 0 and x > 0 and (x_p, y_p) == (x - 1, y))
            or (a == 1 and y > 0 and (x_p, y_p) == (x, y - 1))
            or (a == 2 and x < x_last and (x_p, y_p) == (x + 1, y))
            or (a == 3 and y < y_last and (x_p, y_p) == (x, y + 1))):
            
            if (((x_p, y_p) == (x_last, 0) and r == -3)
                or ((x_p, y_p) == (x_last, y_last) and r == 1)):
                return 1.0
            elif r == 0 and (x_p, y_p) != (x_last, 0) and (x_p, y_p) != (x_last, y_last):
                return 1.0
        if (r == 0 and ((a == 0 and x == 0 and (x_p, y_p) == (0, y))
            or (a == 1 and y == 0 and (x_p, y_p) == (x, 0))
            or (a == 2 and x == x_last and (x_p, y_p) == (x_last, y))
            or (a == 3 and y == y_last and (x_p, y_p) == (x, y_last)))):
            return 1.0
        return 0.0

    def current_state(self) -> int:
        return self.pos[0] + self.pos[1] * self.grid_width

    def reset(self):
        self.pos = (0, 0)
    
    def pretty_print(self):
        for h in range(self.grid_height):
            for w in range(self.grid_width):
                if self.pos == (w, h):
                    print("|X", end="")
                else:
                    print("|_", end="")
            print("|")
    
    def is_forbidden(self, action: int) -> int:
        return False
    
    def is_game_over(self) -> bool:
        return self.pos == (self.grid_width - 1, 0) or self.pos == (self.grid_width - 1, self.grid_height - 1)
    
    def available_actions(self) -> np.ndarray:
        return [0, 1, 2, 3]
    
    def action_desc(self):
        return {0: "Left", 1: "Up", 2: "Right", 3: "Down"}
    
    def step(self, action: int):
        if action not in self.available_actions():
            raise Exception("Action not allowed in current state")
        
        if self.is_game_over():
            raise Exception("Game is already over")
        
        if (action == 0 and self.pos[0] > 0):
            self.pos = (self.pos[0] - 1, self.pos[1])
        if (action == 1 and self.pos[1] > 0):
            self.pos = (self.pos[0], self.pos[1] - 1)
        if (action == 2 and self.pos[0] < self.grid_width - 1):
            self.pos = (self.pos[0] + 1, self.pos[1])
        if (action == 3 and self.pos[1] < self.grid_height - 1):
            self.pos = (self.pos[0], self.pos[1] + 1)
    
    def score(self):
        if self.pos == (self.grid_width - 1, 0):
            return -3
        elif self.pos == (self.grid_width - 1, self.grid_height - 1):
            return 1
        else:
            return 0
    
    @staticmethod
    def from_random_state(grid_width: int = 5, grid_height: int = 5) -> 'GridWorld':
        instance = GridWorld(grid_width, grid_height)
        x = np.random.choice(range(0, grid_width))
        y = np.random.choice(range(0, grid_height))
        while (x, y) == (grid_width - 1, 0) or (x, y) == (grid_width - 1, grid_height - 1):
            x = np.random.choice(range(0, grid_width))
            y = np.random.choice(range(0, grid_height))
        instance.pos = (x, y)
        return instance