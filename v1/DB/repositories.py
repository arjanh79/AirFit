
import uuid
import time


class GenericRepository:
    def __init__(self, db):
        self.db = db  # 


    def get_all_workouts(self):
        sql = """
                SELECT W.w_id, E.e_id, E.weight, E.reps, E.e_sequence
                    FROM WorkoutExercise E
                    JOIN Workout W on W.w_id = E.w_id
                WHERE w.intensity IS NOT NULL
                ORDER BY W.timestamp, W.w_id, E.e_sequence;
              """
        return self.db.execute_query(sql)


    def get_intensities(self):
        sql = 'SELECT w_id, intensity FROM Workout WHERE intensity IS NOT NULL ORDER BY timestamp, w_id;'
        return self.db.execute_query(sql)


    def get_mapping(self):
        sql = 'SELECT min(e_id) mapid, name FROM Exercise GROUP BY name ORDER BY e_id;'
        return self.db.execute_query(sql)


    def delete_unrated_workouts(self):
        unrated_workouts = self.db.execute_query('SELECT w_id FROM Workout WHERE intensity IS NULL;')
        if unrated_workouts:
            delete_exercises = 'DELETE FROM WorkoutExercise WHERE w_id IN (SELECT w_id FROM Workout WHERE intensity IS NULL);'
            delete_workouts = 'DELETE FROM Workout WHERE intensity IS NULL;'
            self.db.execute_query_commit(delete_exercises)
            self.db.execute_query_commit(delete_workouts)


    def get_all_exercises(self):
        sql = 'SELECT name, weight FROM Exercise GROUP BY name, weight ORDER by name, weight'
        return self.db.execute_query(sql)


    def save_workout(self, workout):
        w_id = str(uuid.uuid4())
        now = int(time.time())
        self.new_wo_insert(w_id, now)
        self.new_wo_insert_row(w_id, workout)


    def new_wo_insert_row(self, w_id, wo):
        for row in wo.itertuples(index=False):
            sql = 'INSERT INTO WorkoutExercise(w_id, e_id, e_sequence, weight, reps) VALUES (?, ?, ?, ?, ?)'
            self.db.execute_insert(sql, (w_id, row[0], row[3], row[1], row[2]))


    def new_wo_insert(self, w_id, now):
        sql = 'INSERT INTO Workout(w_id, timestamp) VALUES (?, ?)'
        self.db.execute_insert(sql, (w_id, now))


    def get_available_workout(self):
        sql = ('SELECT W.w_id, E.name, WE.weight, WE.reps, WE.e_sequence '
                'FROM WorkoutExercise WE '
                'JOIN Workout W on W.w_id = WE.w_id '
                'JOIN Exercise E on E.e_id = WE.e_id '
                'WHERE W.intensity IS NULL '
               'ORDER BY WE.e_sequence')
        return self.db.execute_query(sql)


    def save_workout_intensity(self, w_id, intensity):
        now = int(time.time())
        sql = 'UPDATE Workout SET intensity = (?), timestamp = (?) WHERE w_id = (?);'
        self.db.execute_insert(sql, (intensity, now, w_id))




class SQLiteRepository(GenericRepository):
    def __init__(self, db):
        super().__init__(db)
