class GenericRepository:
    def __init__(self, db):
        self.db = db

    def get_all_blocks(self, core):
        sql = 'SELECT block_id FROM Block WHERE core = ?;'
        return self.db.execute_query(sql, (core, ))

    def get_block(self, block_id):
        sql = 'SELECT exercise_id FROM BlockExercise WHERE block_id = ? ORDER BY seq;'
        return self.db.execute_query(sql, (block_id, ))




class SQLiteRepository(GenericRepository):
    def __init__(self, db):
        super().__init__(db)

