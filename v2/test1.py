from v2.Database.factories import RepositoryFactory

repo = RepositoryFactory.get_repository('sqlite')
print(repo.get_all_warming_ups())