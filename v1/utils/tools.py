from datetime import datetime
from v1.DB.factories import RepositoryFactory


def get_workout_date():
    repo = RepositoryFactory.get_repository('sqlite')
    workout_timestamp = repo.get_available_workout_dow()

    if workout_timestamp == -1:
        return False

    today_dow = datetime.now().weekday()
    workout_dow = datetime.fromtimestamp(workout_timestamp).weekday()

    return workout_dow == today_dow
