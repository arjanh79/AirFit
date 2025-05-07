from dev.v1.DB.factories import RepositoryFactory

repo = RepositoryFactory.get_repository('sqlite')
db_mappings = repo.get_mapping()[0]
all_exercises = repo.get_all_exercises()

print(all_exercises)
print('-')
print(db_mappings)