import random
from dataclasses import dataclass
from collections import defaultdict, Counter

from v2.Database.factories import RepositoryFactory

import pandas as pd

@dataclass
class Block:
    block_type: int
    block_id: int
    exercises: list[int]

@dataclass
class Weight:
    block_type: int
    weights: dict[int, int]


class WorkoutGenerator:
    def __init__(self):
        self.repo = RepositoryFactory.get_repository('sqlite')

        # self.repo.delete_unrated_workouts()

        if not self.has_existing_workout():
            self.workout = self.get_blocks()
            self.weights = self.get_exercise_weights()
            self.complete_workout = self.compose()
            self.save_basic_workout()

        self.print_workout()


    def has_existing_workout(self):
        return bool(self.repo.get_existing_workout())

    def print_workout(self):
        print(self.get_clean_workout())

    def get_clean_workout(self):
        workout = self.repo.get_workout()
        df = pd.DataFrame(data=workout[0], columns=workout[1])
        df.columns = ['block', 'seq', 'exercise', 'weight', 'reps', 'equipment']
        return df


    def save_basic_workout(self):
        self.repo.save_workout(self.complete_workout)

    def compose(self):
        combined_workout = []

        for i in [0, 1]:
            block_id = self.workout[i].block_id
            where_pairs = list(self.weights[i].weights.items())
            result = self.repo.get_workout_compose(block_id, where_pairs)
            df = pd.DataFrame(data=result[0], columns=result[1])
            combined_workout.append(df)

        return pd.concat(combined_workout, ignore_index=True)


    def get_block(self, block_type, block_id=None):

        if not block_id:
            block_ids, _ = self.repo.get_all_blocks(block_type)
            block_id = random.choice(block_ids)[0]


        block, _ = self.repo.get_block(block_id)

        exercise_ids = [exercise_id[0] for exercise_id in block]
        return Block(block_type, block_id, exercise_ids)

    def get_blocks(self)    :
        return [
            self.get_block(block_type=0, block_id=5),
            self.get_block(block_type=1, block_id=None)
        ]


    def get_exercise_weights(self):
        return [
            self.get_exercise_weight(0),
            self.get_exercise_weight(1),
        ]

    def get_exercise_weight(self, block_type):

        exercise_ids = self.workout[block_type].exercises

        result, _ = self.repo.get_exercises(exercise_ids)
        weights = defaultdict(set)


        for exercise, weight in result:
            weights[exercise].add(weight)

        while True:
            freq = Counter()
            for weight_set in (v for v in weights.values() if len(v) > 1):
                for w in weight_set:
                    freq[w] += 1

            if not freq:
                break

            max_count = max(freq.values())
            selected_weight = random.choice([w for w, c in freq.items() if c == max_count])

            for ex_id, weight_set in weights.items():
                if len(weight_set) > 1 and selected_weight in weight_set:
                    weights[ex_id] = {selected_weight}

        for ex_id, weight_set in weights.items():
            if len(weight_set) > 1:
                weights[ex_id] = {random.choice(list(weight_set))}

        weights = {k: v.pop() for k, v in weights.items()}
        return Weight(block_type, weights)
