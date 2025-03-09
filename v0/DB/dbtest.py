from dev.v0.DB.factories import RepositoryFactory

repo = RepositoryFactory.get_repository('sqlite')

print(repo.get_intensities())