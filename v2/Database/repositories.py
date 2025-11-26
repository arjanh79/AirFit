class GenericRepository:
    def __init__(self, db):
        self.db = db

    def get_all_warming_ups(self):
        sql = 'SELECT block_id FROM Block WHERE core = 0;'
        print(sql)
        return self.db.execute_query(sql)


class SQLiteRepository(GenericRepository):
    def __init__(self, db):
        super().__init__(db)

