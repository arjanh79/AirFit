
import pandas as pd
import string
import random

from v2.Data.factories import RepositoryFactory

class NewWorkoutBlocks:
    def __init__(self):
        self.repo = RepositoryFactory.get_repository('sqlite')
        self.existing_blocks = self.get_existing_blocks()
        self.new_blocks, self.new_blocks_grouped = self.get_new_blocks()
        self.found_blocks = self.search_new_blocks()

    def get_existing_blocks(self):
        data, cols = self.repo.get_block_workouts()
        df = pd.DataFrame(data, columns=cols)
        df = df[['block_id', 'exercise_id', 'core']]
        df = df.groupby(['block_id', 'core'])['exercise_id'].apply(list).reset_index(drop=False)
        df = df[['core', 'exercise_id']]
        df = df.groupby(['core'])['exercise_id'].apply(list).reset_index(drop=False)
        return df


    def get_new_blocks(self):
        data, cols = self.repo.get_new_blocks()
        df = pd.DataFrame(data, columns=cols)
        grouped = df.groupby(['workout_id', 'core'])['exercise_id'].apply(list).reset_index(drop=False)
        return df, grouped


    def search_new_blocks(self):

        blocks = []
        for block in [0, 1]:
            df_temp = self.existing_blocks[self.existing_blocks['core'] == block]
            blocks.append([set(exercises) for exercises in df_temp['exercise_id'].values[0]])

        new_blocks = set()
        for row in self.new_blocks_grouped.itertuples(name=None, index=False):
            workout_id, core, exercises = row
            if not set(exercises) in blocks[core]:
                new_blocks.add((workout_id, core))

        return new_blocks


    def create_sql(self):
        chars = string.ascii_uppercase + string.ascii_lowercase + string.digits
        for workout_id, core in self.found_blocks:
            block_name = 'AI_' + ''.join(random.sample(chars, 8))
            new_block = self.new_blocks[(self.new_blocks['workout_id'] == workout_id) & (self.new_blocks['core'] == core)]

            data, _ = self.repo.create_new_block(block_name, core)
            block_id = data[0][0]

            for block in new_block.itertuples(name=None, index=False):
                self.repo.add_exercise_block(block_id, block[3], block[1])


nwb = NewWorkoutBlocks()
nwb.create_sql()