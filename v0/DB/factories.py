import sqlite3

class DatabaseFactory:

    @staticmethod
    def get_database(db_type='sqlite', **kwargs):
        if db_type ==  'sqlite':
            return SQLiteDB(kwargs.get('db_name'))
        else:
            raise ValueError(f'UNKNOWN: {db_type=}')


class GenericDB:
    def __init__(self, db_name, db_type):
        self.db_name = db_name
        self.db_type = db_type

    def execute_query(self, sql, params=None):
        with self.db_type.connect(self.db_name) as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(sql, params or ())
                return cursor.fetchall()
            finally:
                cursor.close()

    def execute_query_commit(self, sql, params=None):
        with self.db_type.connect(self.db_name) as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(sql, params or ())
                conn.commit()
            except Exception as e:
                conn.rollback()
                raise e
            finally:
                cursor.close()


class SQLiteDB(GenericDB):
    def __init__(self, db_name):
        super().__init__(db_name, sqlite3)

def empty():
    pass