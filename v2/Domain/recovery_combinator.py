import pandas as pd


from v2.Data.factories import RepositoryFactory


class RecoveryCombinator:
    def __init__(self):
        self.repository = RepositoryFactory.get_repository('sqlite')

    def get_data(self):
        data, cols = self.repository.get_recovery_data()
        df = pd.DataFrame(data, columns=cols)
        df = df.sort_values(['timestamp', 'core' ,'exercise_sequence'])

        agg = {col: list for col in df.columns if col not in ['workout_id', 'timestamp', 'train_tomorrow']}
        df = df.groupby(['workout_id', 'train_tomorrow'], as_index=False).agg(agg)

        return df
