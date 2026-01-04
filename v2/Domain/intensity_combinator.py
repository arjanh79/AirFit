import pandas as pd


from v2.Data.factories import RepositoryFactory


class IntensityCombinator:
    def __init__(self):
        self.repository = RepositoryFactory.get_repository('sqlite')

    def get_data(self, completed):
        data, cols = self.repository.get_training_data(completed = completed)
        df = pd.DataFrame(data, columns=cols)
        df = df.sort_values(['timestamp', 'core' ,'exercise_sequence'])

        if completed:
            agg = {col: list for col in df.columns if col not in ['workout_id', 'timestamp', 'workout_intensity']}
            df = df.groupby(['workout_id', 'workout_intensity'], as_index=False).agg(agg)

        return df
