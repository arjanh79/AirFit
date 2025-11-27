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
               f'WHERE EW.exercise_id IN ({params}) '
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


class SQLiteRepository(GenericRepository):
    def __init__(self, db):
        super().__init__(db)

