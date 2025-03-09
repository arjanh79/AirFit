from dev.v0.DB.databases import SQLiteDB
from dev.v0.DB.repositories import SQLiteRepository


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
        db = DatabaseFactory.get_database(db_type, db_name='airfit.db', **kwargs)

        if db_type == 'sqlite':
            return SQLiteRepository(db)
        else:
            raise ValueError(f'UNKNOWN: {db_type=}')
