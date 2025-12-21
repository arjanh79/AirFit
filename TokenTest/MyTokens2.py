from collections import defaultdict
from dataclasses import dataclass
import pandas as pd
from typing import DefaultDict

from v2.Database.factories import RepositoryFactory

@dataclass
class Block:
    block_type: int
    block_id: int
    exercises: set[int]

class MyTokens2:
    def __init__(self):
        self.repo = RepositoryFactory.get_repository('sqlite')

        self.df = self.get_dataset()

        # Also do some data verification in this step.
        try:
            self.a_blocks, self.b_blocks = self.get_blocks()
        except ValueError as e:
            print(e)

        self.allowed_combinations = self.match_blocks()
        self.workouts = self.create_workouts()


    def get_dataset(self) -> pd.DataFrame:
        data_rows, col_names = self.repo.get_block_workouts()
        return pd.DataFrame(data=data_rows, columns=col_names)


    def get_blocks(self) -> tuple[list[Block], list[Block]]:
        grouped = (
            self.df[['block_id', 'core', 'exercise_id']]
            .groupby(['block_id', 'core'])['exercise_id']
            .apply(list)
            .reset_index()
        )

        a_blocks: list[Block] = []
        b_blocks: list[Block] = []

        seen_exercise_sets: dict[frozenset[int], int] = {}


        for row in grouped.itertuples(index=False):
            block = Block(block_type=int(row.core),
                          block_id=int(row.block_id),
                          exercises={int(x) for x in row.exercise_id})

            if len(block.exercises) != 6:
                raise ValueError(f'Block {block.block_id} should have 6 unique exercises, found {len(block.exercises)}.')

            exercises = frozenset(block.exercises)
            if exercises in seen_exercise_sets:
                raise ValueError(f'Duplicate exercises {sorted(exercises)} in block {block.block_id}.'
                                 f' This block is a duplicate of block {seen_exercise_sets[exercises]}.')
            else:
                seen_exercise_sets[exercises] = block.block_id

            if block.block_type == 0:
                a_blocks.append(block)
            else:
                b_blocks.append(block)

        return a_blocks, b_blocks


    def match_blocks(self, max_overlap=1) -> dict[int, list[int]]:

        allowed: DefaultDict[int, list[int]] = defaultdict(list)

        for a in self.a_blocks:
            for b in self.b_blocks:
                if len(a.exercises & b.exercises) <= max_overlap:
                    allowed[a.block_id].append(b.block_id)

        return dict(allowed)


    def create_workouts(self) -> list[list[int]]:

        ab_pairs = ((a_id, b_id)
                    for a_id, b_ids in self.allowed_combinations.items()
                    for b_id in b_ids)

        exercises_by_block = (
            self.df.groupby('block_id')['exercise_id']
            .apply(lambda s: [int(x) for x in s])
            .to_dict()
        )

        workouts: list[list[int]] = []
        for a_id, b_id in ab_pairs:
            workouts.append(exercises_by_block[a_id] + exercises_by_block[b_id])

        return workouts



mt2 = MyTokens2()
