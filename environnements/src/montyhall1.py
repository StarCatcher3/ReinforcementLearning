from .baseenv import EnvTemplate
import numpy as np


class MontyHallLevel1(EnvTemplate):

    def __init__(self):
        self.winning_door = None
        self.stage = 0
        self.first_choice = None
        self.removed_door = None
        self.final_choice = None
        self.won = None
        self.reset()

    def maximum_states_count(self) -> int:
        return 4

    def maximum_actions_count(self) -> int:
        return 3

    def num_rewards(self) -> int:
        return 2

    def reward(self, i: int) -> float:
        return [0.0, 1.0][i]

    def p(self, s: int, a: int, s_p: int, r_index: int) -> float:
        if s == 0:
            if a in (0, 1, 2) and s_p == 1 and r_index == 0:
                return 1.0
            return 0.0
        if s == 1:
            if a == 0:
                if s_p == 3 and r_index == 1:
                    return 1 / 3
                if s_p == 2 and r_index == 0:
                    return 2 / 3
            elif a == 1:
                if s_p == 3 and r_index == 1:
                    return 2 / 3
                if s_p == 2 and r_index == 0:
                    return 1 / 3
            return 0.0
        return 0.0

    def current_state(self) -> int:
        if self.stage == 0:
            return 0
        if self.stage == 1:
            return 1
        return 3 if self.won else 2

    def reset(self):
        self.winning_door = np.random.choice([0, 1, 2])
        self.stage = 0
        self.first_choice = None
        self.removed_door = None
        self.final_choice = None
        self.won = None

    @staticmethod
    def _label(door: int) -> str:
        return chr(ord("A") + door)

    def pretty_print(self):
        if self.stage == 0:
            print("3 portes fermées : A, B, C. Choisissez-en une.")
        elif self.stage == 1:
            print(f"Vous avez choisi la porte {self._label(self.first_choice)}. "
                  f"La porte {self._label(self.removed_door)} est ouverte : elle ne contient rien. "
                  "Voulez-vous garder votre porte (0) ou changer (1) ?")
        else:
            print(f"La porte {self._label(self.final_choice)} est ouverte : "
                  f"{'GAGNÉ' if self.won else 'PERDU'} (la porte gagnante était {self._label(self.winning_door)}).")

    def is_forbidden(self, action: int) -> int:
        return action not in self.available_actions()

    def is_game_over(self) -> bool:
        return self.stage == 2

    def available_actions(self) -> np.ndarray:
        if self.stage == 0:
            return [0, 1, 2]
        if self.stage == 1:
            return [0, 1]
        return []

    def action_desc(self):
        if self.stage == 0:
            return {0: "Choisir la porte A", 1: "Choisir la porte B", 2: "Choisir la porte C"}
        return {0: "Garder sa porte", 1: "Changer de porte"}

    def step(self, action: int):
        if action not in self.available_actions():
            raise Exception("Action not allowed in current state")

        if self.is_game_over():
            raise Exception("Game is already over")

        if self.stage == 0:
            self.first_choice = action
            others = [d for d in range(3) if d != self.first_choice]
            non_winning_others = [d for d in others if d != self.winning_door]
            if len(non_winning_others) == 2:
                self.removed_door = np.random.choice(non_winning_others)
            else:
                self.removed_door = non_winning_others[0]
            self.stage = 1
        elif self.stage == 1:
            if action == 0:
                self.final_choice = self.first_choice
            else:
                self.final_choice = [d for d in range(3) if d != self.first_choice and d != self.removed_door][0]
            self.won = (self.final_choice == self.winning_door)
            self.stage = 2

    def score(self):
        if self.stage == 2:
            return 1.0 if self.won else 0.0
        return 0.0

    @staticmethod
    def from_random_state() -> 'MontyHallLevel1':
        instance = MontyHallLevel1()
        if np.random.rand() < 0.5:
            return instance
        first_choice = np.random.choice([0, 1, 2])
        instance.step(first_choice)
        return instance
