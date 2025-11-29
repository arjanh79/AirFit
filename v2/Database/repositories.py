import uuid
import time

class GenericRepository:
    def __init__(self, db):
        self.db = db

    def get_all_blocks(self, core):
        sql = 'SELECT block_id FROM Block WHERE core = ?;'
        return self.db.execute_query(sql, (core, ))

    def get_block(self, block_id):
        sql = 'SELECT exercise_id FROM BlockExercise WHERE block_id = ? ORDER BY seq;'
        return self.db.execute_query(sql, (block_id, ))

    def get_exercises(self, exercise_ids):
        params = ','.join(['?' for _ in exercise_ids])

        sql = (f'SELECT EW.exercise_id, EW.weight_id FROM ExerciseWeight EW '
               f'JOIN Weight W on EW.weight_id = W.weight_id '
               f'JOIN Exercise E on EW.exercise_id = E.exercise_id '
               f'WHERE EW.available=1 AND EW.exercise_id IN ({params}) '
               f'GROUP BY EW.exercise_id, EW.weight_id')

        return self.db.execute_query(sql, exercise_ids)

    def get_workout_compose(self, block_id, where_pairs):

        conditions = " OR ".join(
            "(BE.exercise_id = ? AND EW.weight_id = ?)"
            for _ in where_pairs
        )

        sql = (f'SELECT B.block_id, B.core, BE.seq, BE.exercise_id, E.default_value, W.weight_id, EQ.equipment_id '
               f'FROM BlockExercise BE '
               f'JOIN Block B ON BE.block_id = B.block_id '
               f'JOIN Exercise E ON BE.exercise_id = E.exercise_id '
               f'JOIN ExerciseWeight EW ON BE.exercise_id = EW.exercise_id '
               f'JOIN Weight W ON EW.weight_id = W.weight_id '
               f'JOIN Equipment EQ on W.equipment_id = EQ.equipment_id '
               f'WHERE B.block_id = ? AND ({conditions}) ORDER BY seq')

        where_pairs = [item for i in where_pairs for item in i]
        where_pairs = [block_id] + where_pairs

        return self.db.execute_query(sql,where_pairs)

    def save_workout(self, workout):
        self.delete_unrated_workouts()

        workout_id = str(uuid.uuid4())

        self.save_workout_header(workout_id)
        self.save_workout_body(workout, workout_id)


    def save_workout_header(self, workout_id):
        now = int(time.time())
        sql = 'INSERT INTO Workout(workout_id, timestamp) VALUES (?, ?)'
        self.db.execute_insert(sql, (workout_id, now))


    def save_workout_body(self, workout, workout_id):
        for row in workout.itertuples(index=False):
            sql = 'INSERT INTO WorkoutExercise(workout_id, exercise_id, exercise_sequence, weight_id, reps, block_id, core, equipment_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?)'
            self.db.execute_insert(sql, (workout_id, row[3], row[2], row[5], row[4], row[0], row[1], row[6]))


    def delete_unrated_workouts(self):
        unrated_workouts = self.db.execute_query('SELECT workout_id FROM Workout WHERE workout_intensity IS NULL;')
        if unrated_workouts:
            delete_exercises = 'DELETE FROM WorkoutExercise WHERE workout_id IN (SELECT workout_id FROM Workout WHERE workout_intensity IS NULL);'
            delete_workouts = 'DELETE FROM Workout WHERE workout_intensity IS NULL;'
            self.db.execute_query_commit(delete_exercises)
            self.db.execute_query_commit(delete_workouts)


    def get_workout(self):
        sql = ('SELECT WE.core, WE.exercise_sequence, EX.name, K.weight, WE.reps, E.name '
               'FROM Workout W '
               'JOIN WorkoutExercise WE on W.workout_id = WE.workout_id '
               'JOIN Exercise EX on EX.exercise_id = WE.exercise_id '
               'JOIN Weight K on K.weight_id = WE.weight_id '
               'JOIN Equipment E on E.equipment_id = WE.equipment_id '
               'WHERE W.workout_intensity IS NULL;')
        return self.db.execute_query(sql)

    def get_existing_workout(self):
        unrated_workouts = self.db.execute_query('SELECT workout_id FROM Workout WHERE workout_intensity IS NULL;')
        return len(unrated_workouts[0])


class SQLiteRepository(GenericRepository):
    def __init__(self, db):
        super().__init__(db)

