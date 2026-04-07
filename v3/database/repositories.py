
class GenericRepository:
    def __init__(self, db):
        self.db = db

    def get_training_data(self):
        sql = '''SELECT W.workout_id, W.workout_intensity, W.train_tomorrow, W.active_kcal, 
                WE.exercise_id, WE.exercise_sequence, WE.core, WE.reps, WE.weight_id
                FROM Workout W 
                JOIN WorkoutExercise WE ON W.workout_id = WE.workout_id
                WHERE W.workout_completed >= 1773763768 AND W.workout_intensity NOT NULL
                ORDER BY W.workout_completed DESC, WE.core, WE.exercise_sequence'''

        return self.db.execute_query(sql)


class SQLiteRepository(GenericRepository):
    def __init__(self, db):
        super().__init__(db)
