from dataclasses import dataclass

from v2.Utils.exercise_id_mapping import Mappings


@dataclass
class PairRule:
    a: int
    b: int

    def bi_directional(self) -> tuple[tuple[int, int], tuple[int, int]]:
        return (self.a, self.b), (self.b, self.a)


class BlockedTokens:
    def __init__(self):

        not_next = [('Deadlift', 'Step Ups'),
                    ('Bosu Plank', 'Shoulder Taps'),
                    ('Shoulder Taps', 'Plank'),
                    ('Bosu Plank', 'Bosu Mountain Climbers'),
                    ('Split Squats', 'Deadlift'),
                    ('Dumbbell Press', 'Bosu Push Up'),
                    ('Plank', 'Bosu Push Up'),
                    ('Shoulder Taps', 'Bosu Push Up'),
                    ('Squats', 'Step Ups'),
                    ('Clean and Press', 'Dumbbell Snatch'),
                    ('Reverse Fly', 'Bent-Over Row'),
                    ('Clean and Press', 'Deadlift'),
                    ('Clean and Press', 'Bosu Clean and Press'),
                    ('Split Squats', 'Step Ups')]

        not_same = [('Bosu Plank', 'Plank')]

        not_start = ['Clean and Press', 'Deadlift', 'Dumbbell Snatch', 'Split Squats']

        self.exercise_tokens = Mappings().exercise_to_token  # All known mappings

        self.not_next = self.get_pairs((self.get_tokens(not_next)))
        self.not_same = self.get_pairs((self.get_tokens(not_same)))
        self.not_start = self.get_start_tokens(not_start)


    def get_pairs(self, rule):
        return {(x, y) for a, b in rule for x, y in PairRule(a, b).bi_directional()}


    def get_tokens(self, pairs):
        return [(self.exercise_tokens[a], self.exercise_tokens[b]) for a, b in pairs]


    def get_start_tokens(self, exercises):
        return {self.exercise_tokens[name] for name in exercises}


    def get_blocked_tokens(self, tokens: list[int]) -> list[int]:
        # tokens always start with START_TOKEN 1

        if len(tokens) == 1:
            return list(self.not_start)

        next_blocked = {b for a, b in self.not_next if a == tokens[-1]}
        same_blocked = {b for a, b in self.not_same if a in tokens}
        return list(next_blocked | same_blocked)
