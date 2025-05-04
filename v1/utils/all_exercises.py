from dev.v1.DB.factories import RepositoryFactory

repo = RepositoryFactory.get_repository('sqlite')
db_mappings = repo.get_mapping()[0]

exercises = [e[1] for e in db_mappings]

print(exercises)