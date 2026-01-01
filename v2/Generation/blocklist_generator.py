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
                    ('Static Lunge', 'Deadlift')]

        not_same = [('Bosu Plank', 'Plank')]

        self.exercise_tokens = Mappings().exercise_to_token  # All known mappings

        self.not_next = self.get_pairs((self.get_tokens(not_next)))
        self.not_same = self.get_pairs((self.get_tokens(not_same)))


    def get_pairs(self, rule):
        return {(x, y) for a, b in rule for x, y in PairRule(a, b).bi_directional()}


    def get_tokens(self, pairs):
        return [(self.exercise_tokens[a], self.exercise_tokens[b]) for a, b in pairs]


    def get_blocked_tokens(self, tokens: list[int]) -> list[int]:
        # tokens always start with START_TOKEN 1
        next_blocked = {b for a, b in self.not_next if a == tokens[-1]}
        same_blocked = {b for a, b in self.not_same if a in tokens}
        return list(next_blocked | same_blocked)


