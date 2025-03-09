
class GenericRepository:
    def __init__(self, db):
        self.db = db  # 

    def get_all_workouts(self):
        sql = """
                SELECT W.w_id, E.e_id, E.reps, E.weight, E.e_sequence
                    FROM WorkoutExercise E
                    JOIN Workout W on W.w_id = E.w_id
                WHERE w.intensity IS NOT NULL
                ORDER BY W.timestamp, E.e_sequence;
              """
        return self.db.execute_query(sql)

    def get_intensities(self):
        sql = 'SELECT w_id, intensity FROM Workout WHERE intensity IS NOT NULL ORDER BY timestamp;'
        return self.db.execute_query(sql)

    def get_mapping(self):
        sql = 'SELECT min(e_id) mapid, name FROM Exercise GROUP BY name'
        return self.db.execute_query(sql)

    def delete_unrated_workouts(self):
        unrated_workouts = self.db.execute_query('SELECT w_id FROM Workout WHERE intensity IS NULL;')
        if unrated_workouts:
            delete_exercises = 'DELETE FROM WorkoutExercise WHERE w_id IN (SELECT w_id FROM Workout WHERE intensity IS NULL);'
            delete_workouts = 'DELETE FROM Workout WHERE intensity IS NULL;'
            self.db.execute_query_commit(delete_exercises)
            self.db.execute_query_commit(delete_workouts)

class SQLiteRepository(GenericRepository):
    def __init__(self, db):
        super().__init__(db)
