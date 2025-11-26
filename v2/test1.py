import random

from v2.Database.factories import RepositoryFactory

repo = RepositoryFactory.get_repository('sqlite')
warming_up_list = repo.get_all_warming_ups()[0]
core_list = repo.get_all_cores()[0]

warming_up_id = random.choice(warming_up_list)[0]
core_id = random.choice(core_list)[0]

warming_up = repo.get_block(warming_up_id)
core = repo.get_block(core_id)

print(warming_up)
print(core)