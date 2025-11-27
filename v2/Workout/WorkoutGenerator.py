import random
from dataclasses import dataclass

from v2.Database.factories import RepositoryFactory


@dataclass
class Block:
    block_type: int
    block_id: int
    exercises: list[int]


class WorkoutGenerator:
    def __init__(self):
        self.repo = RepositoryFactory.get_repository('sqlite')
        self.workout = self.get_blocks()
        print(self.workout[0])
        print(self.workout[1])

    def get_block(self, block_type: int):
        block_ids, _ = self.repo.get_all_blocks(block_type)
        block_id = random.choice(block_ids)[0]
        block, _ = self.repo.get_block(block_id)

        exercise_ids = [exercise_id[0] for exercise_id in block]
        return Block(block_type, block_id, exercise_ids)

    def get_blocks(self)    :
        return [
            self.get_block(0),
            self.get_block(1)
        ]


wg = WorkoutGenerator()