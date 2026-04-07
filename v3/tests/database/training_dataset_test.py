from v3.database.factories import RepositoryFactory

if __name__ == '__main__':
    repo = RepositoryFactory.get_repository('sqlite')
    training_data = repo.get_training_data()
    print(f'Columns: {training_data[1]}')
    for row in training_data[0]:
        print(row)