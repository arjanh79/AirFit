import uuid
import time

class GenericRepository:
    def __init__(self, db):
        self.db = db


    def get_exercises(self, exercise_ids):
        params = ','.join(['?' for _ in exercise_ids])

        sql = (f'SELECT EW.exercise_id, EW.weight_id FROM ExerciseWeight EW '
               f'JOIN Weight W on EW.weight_id = W.weight_id '
               f'JOIN Exercise E on EW.exercise_id = E.exercise_id '
               f'WHERE EW.available=1 AND EW.exercise_id IN ({params}) '
               f'GROUP BY EW.exercise_id, EW.weight_id')

        return self.db.execute_query(sql, exercise_ids)


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
            sql = 'INSERT INTO WorkoutExercise(workout_id, exercise_id, exercise_sequence, weight_id, reps, core, equipment_id) VALUES (?, ?, ?, ?, ?, ?, ?)'
            self.db.execute_insert(sql, (workout_id, row[0], row[1], row[2], row[3], row[4], row[5]))


    def get_workout_details(self, workout: list[tuple[int, int]]):
        placeholders = ", ".join(["(?, ?)"] * len(workout))
        condition = f"(E.exercise_id, W.weight_id) IN ({placeholders})"
        params = [value for pair in workout for value in pair]

        sql = (f'SELECT EW.exercise_id, W.weight_id, E.default_value, W.equipment_id FROM ExerciseWeight EW ' 
               f'JOIN Exercise E on E.exercise_id = EW.exercise_id '
               f'JOIN Weight W on W.weight_id = EW.weight_id WHERE {condition}')

        return self.db.execute_query(sql, params)



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


    def get_exercise_ids(self):
        sql = 'SELECT exercise_id, name FROM Exercise;'
        return self.db.execute_query(sql)


    def get_weight_ids(self):
        sql = 'SELECT weight_id FROM Weight;'
        return self.db.execute_query(sql)


    def get_equipment_ids(self):
        sql = 'SELECT equipment_id FROM Equipment;'
        return self.db.execute_query(sql)


    def get_block_workouts(self):
        sql = ('SELECT BE.block_id, BE.seq, B.description, E.exercise_id, E.name, B.core '
               'FROM BlockExercise BE '
               'JOIN Block B on B.block_id = BE.block_id '
               'JOIN Exercise E on E.exercise_id = BE.exercise_id '
               'ORDER BY BE.block_id, seq;')
        return self.db.execute_query(sql)


    def get_new_blocks(self):
        sql = ('SELECT WE.workout_id, WE.exercise_id, WE.core, WE.exercise_sequence '
               'FROM WorkoutExercise WE '
               'JOIN Workout W ON W.workout_id = WE.workout_id '
               'WHERE W.workout_intensity NOT NULL')
        return self.db.execute_query(sql)

    def check_available_workout(self):
        sql = 'SELECT workout_id FROM Workout WHERE workout_intensity IS NULL;'
        return self.db.execute_query(sql)


    def get_training_data(self):
        sql = ('SELECT W.timestamp, W.workout_id, W.workout_intensity, WE.exercise_id, WE.exercise_sequence, WE.weight_id, WE.reps, WE.core, E.metric_type, G.equipment_id '
               'FROM Workout W '
               'JOIN WorkoutExercise WE on WE.workout_id = W.workout_id '
               'JOIN Exercise E on E.exercise_id = WE.exercise_id '
               'JOIN Weight G on WE.weight_id = G.weight_id '
               'WHERE W.workout_intensity NOT NULL '
               'ORDER BY W.timestamp, WE.core, WE.exercise_sequence')
        return self.db.execute_query(sql)


    def get_all_weights(self):
        sql = ('SELECT W.weight, E.name, W.weight_id '
               'FROM weight W '
               'JOIN equipment E ON E.equipment_id = W.equipment_id '
               'ORDER BY W.weight, E.name;')
        return self.db.execute_query(sql)

class SQLiteRepository(GenericRepository):
    def __init__(self, db):
        super().__init__(db)

