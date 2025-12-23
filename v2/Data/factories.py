from v2.Data.databases import SQLiteDB
from v2.Data.repositories import SQLiteRepository
from v2.config import DB_PATH

class DatabaseFactory:
    @staticmethod
    def get_database(db_type='sqlite', **kwargs):
        if db_type ==  'sqlite':
            return SQLiteDB(kwargs.get('db_name'))
        else:
            raise ValueError(f'UNKNOWN: {db_type=}')


class RepositoryFactory:
    @staticmethod
    def get_repository(db_type='sqlite', **kwargs):
        db = DatabaseFactory.get_database(db_type, db_name=f'{DB_PATH}', **kwargs)

        if db_type == 'sqlite':
            return SQLiteRepository(db)
        else:
            raise ValueError(f'UNKNOWN: {db_type=}')
