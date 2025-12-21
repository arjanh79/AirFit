
from v2.Database.factories import RepositoryFactory
import string
import random


class NewWorkout:

    def __init__(self, filename):
        self.repo = RepositoryFactory.get_repository('sqlite')

        self.filename = filename
        self.workouts = self.get_new_workouts()

        self.id_to_name, self.name_to_id = self.get_exercise_ids()
        self.exercise_ids = self.workouts_to_ids()


    def get_new_workouts(self):
        with open(self.filename, 'r') as f:
            lines = [line.rstrip().split(',') for line in f]
        lines = [[exercise.strip() for exercise in line] for line in lines]
        return lines


    def get_exercise_ids(self):
        result = self.repo.get_exercise_ids()[0]
        ids, names = zip(*result)
        return dict(zip(ids, names)), dict(zip(names, ids))


    def workouts_to_ids(self):
        return [[self.name_to_id[exercise] for exercise in block] for block in self.workouts]


    def generate_sql(self):

        chars = string.ascii_uppercase + string.ascii_lowercase + string.digits

        for c, block in enumerate(self.exercise_ids):
            name = 'AI_' + ''.join(random.choices(chars, k=8))
            print(f'''INSERT INTO Block(block_id, description, core) VALUES ({c+17}, '{name}', 1);''')
            for d, exercise_id in enumerate(block):
                print(f'INSERT INTO BlockExercise(block_id, seq, exercise_id) VALUES ({c+17}, {d+1}, {exercise_id});')


nw = NewWorkout('new_workouts')
nw.generate_sql()