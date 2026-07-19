from .baseenv import EnvTemplate
import numpy as np


class MontyHallLevel2(EnvTemplate):
    """Monty Hall paradox, level 2 (5 doors, 4 successive decisions).

    Mechanism (generalizing level 1): the agent first picks 1 of the 5 doors.
    Then, 3 times in a row: the host removes one non-winning door that is not
    the agent's current choice, and the agent decides to stay with its current
    choice or switch to one of the other doors still in play (uniformly at
    random among them, as they are interchangeable by symmetry). After the
    3rd elimination only 2 doors remain (current choice + 1 other); the 4th
    and last decision is applied to those 2 doors and the chosen one is opened.

    States are abstracted (no literal door identity is tracked in current_state(),
    only the game stage and the stay/switch decisions taken so far, since
    those are what determine the probability of the current choice being the
    winning one -- the same symmetry principle already used by
    RockPaperScissors and MontyHallLevel1):
        0 -> before the 1st decision (5 doors)
        1 -> before the 2nd decision (4 doors)
        2 -> before the 3rd decision, having stayed at the 2nd decision
        3 -> before the 3rd decision, having switched at the 2nd decision
        4 -> before the 4th decision, path (stay, stay)
        5 -> before the 4th decision, path (stay, switch)
        6 -> before the 4th decision, path (switch, stay)
        7 -> before the 4th decision, path (switch, switch)
        8 -> game over, lost (reward 0.0)
        9 -> game over, won (reward 1.0)

    Actions:
        at state 0: 0/1/2/3/4 -> choose door A/B/C/D/E
        at states 1, 2, 3, 4, 5, 6, 7: 0 -> stay, 1 -> switch
    """

    # (state, action) -> [(next_state, reward_index, probability), ...]
    _TRANSITIONS = {
        (0, 0): [(1, 0, 1.0)],
        (0, 1): [(1, 0, 1.0)],
        (0, 2): [(1, 0, 1.0)],
        (0, 3): [(1, 0, 1.0)],
        (0, 4): [(1, 0, 1.0)],
        (1, 0): [(2, 0, 1.0)],
        (1, 1): [(3, 0, 1.0)],
        (2, 0): [(4, 0, 1.0)],
        (2, 1): [(5, 0, 1.0)],
        (3, 0): [(6, 0, 1.0)],
        (3, 1): [(7, 0, 1.0)],
        (4, 0): [(9, 1, 1 / 5), (8, 0, 4 / 5)],
        (4, 1): [(9, 1, 4 / 5), (8, 0, 1 / 5)],
        (5, 0): [(9, 1, 2 / 5), (8, 0, 3 / 5)],
        (5, 1): [(9, 1, 3 / 5), (8, 0, 2 / 5)],
        (6, 0): [(9, 1, 4 / 15), (8, 0, 11 / 15)],
        (6, 1): [(9, 1, 11 / 15), (8, 0, 4 / 15)],
        (7, 0): [(9, 1, 11 / 30), (8, 0, 19 / 30)],
        (7, 1): [(9, 1, 19 / 30), (8, 0, 11 / 30)],
    }

    def __init__(self):
        self.num_doors = 5
        self.winning_door = None
        self.current_choice = None
        self.eliminated = set()
        self.round2_action = None
        self.round3_action = None
        self.stage = 0
        self.won = None
        self.reset()

    # MDP related Methods
    def maximum_states_count(self) -> int:
        return 10

    def maximum_actions_count(self) -> int:
        return 5

    def num_rewards(self) -> int:
        return 2

    def reward(self, i: int) -> float:
        return [0.0, 1.0][i]

    def p(self, s: int, a: int, s_p: int, r_index: int) -> float:
        for (next_s, r_idx, prob) in self._TRANSITIONS.get((s, a), []):
            if next_s == s_p and r_idx == r_index:
                return prob
        return 0.0

    # Monte Carlo and TD Methods related functions:
    def current_state(self) -> int:
        if self.stage == 0:
            return 0
        if self.stage == 1:
            return 1
        if self.stage == 2:
            return 2 if self.round2_action == 0 else 3
        if self.stage == 3:
            if self.round2_action == 0 and self.round3_action == 0:
                return 4
            if self.round2_action == 0 and self.round3_action == 1:
                return 5
            if self.round2_action == 1 and self.round3_action == 0:
                return 6
            return 7
        return 9 if self.won else 8

    def reset(self):
        self.winning_door = np.random.choice(range(self.num_doors))
        self.current_choice = None
        self.eliminated = set()
        self.round2_action = None
        self.round3_action = None
        self.stage = 0
        self.won = None

    def _eliminate_one_door(self):
        remaining_non_chosen = [d for d in range(self.num_doors)
                                 if d != self.current_choice and d not in self.eliminated]
        non_winning_remaining = [d for d in remaining_non_chosen if d != self.winning_door]
        if len(non_winning_remaining) > 1:
            self.eliminated.add(np.random.choice(non_winning_remaining))
        else:
            self.eliminated.add(non_winning_remaining[0])

    @staticmethod
    def _label(door: int) -> str:
        return chr(ord("A") + door)

    def pretty_print(self):
        remaining = [d for d in range(self.num_doors) if d not in self.eliminated]
        remaining_str = ", ".join(self._label(d) for d in remaining)
        if self.stage == 0:
            print(f"{self.num_doors} portes fermées : {remaining_str}. Choisissez-en une.")
        elif self.stage < 4:
            print(f"Porte(s) restante(s) : {remaining_str}. Choix actuel : {self._label(self.current_choice)}. "
                  "Garder (0) ou changer (1) ?")
        else:
            print(f"La porte {self._label(self.current_choice)} est ouverte : "
                  f"{'GAGNÉ' if self.won else 'PERDU'} (porte gagnante : {self._label(self.winning_door)}).")

    def is_forbidden(self, action: int) -> int:
        return action not in self.available_actions()

    def is_game_over(self) -> bool:
        return self.stage == 4

    def available_actions(self) -> np.ndarray:
        if self.stage == 0:
            return list(range(self.num_doors))
        if self.stage < 4:
            return [0, 1]
        return []

    def action_desc(self):
        if self.stage == 0:
            return {i: f"Choisir la porte {self._label(i)}" for i in range(self.num_doors)}
        return {0: "Garder sa porte", 1: "Changer de porte"}

    def step(self, action: int):
        if action not in self.available_actions():
            raise Exception("Action not allowed in current state")

        if self.is_game_over():
            raise Exception("Game is already over")

        if self.stage == 0:
            self.current_choice = action
            self._eliminate_one_door()
            self.stage = 1
        elif self.stage == 1:
            self.round2_action = action
            if action == 1:
                candidates = [d for d in range(self.num_doors)
                              if d != self.current_choice and d not in self.eliminated]
                self.current_choice = np.random.choice(candidates)
            self._eliminate_one_door()
            self.stage = 2
        elif self.stage == 2:
            self.round3_action = action
            if action == 1:
                candidates = [d for d in range(self.num_doors)
                              if d != self.current_choice and d not in self.eliminated]
                self.current_choice = np.random.choice(candidates)
            self._eliminate_one_door()
            self.stage = 3
        elif self.stage == 3:
            if action == 1:
                candidates = [d for d in range(self.num_doors)
                              if d != self.current_choice and d not in self.eliminated]
                self.current_choice = candidates[0]
            self.won = (self.current_choice == self.winning_door)
            self.stage = 4

    def score(self):
        if self.stage == 4:
            return 1.0 if self.won else 0.0
        return 0.0

    @staticmethod
    def from_random_state() -> 'MontyHallLevel2':
        instance = MontyHallLevel2()
        target_stage = np.random.choice([0, 1, 2, 3])
        for _ in range(target_stage):
            a = np.random.choice(instance.available_actions())
            instance.step(a)
        return instance
