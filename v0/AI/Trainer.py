import pandas as pd
import torch
from torch.utils.data import Dataset

from dev.v0.DB.factories import RepositoryFactory

class WorkoutDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class DNNTrainer:
    def __init__(self):

        # Load SQL statements.
        self.repo = RepositoryFactory.get_repository('sqlite')

        # Collect all the needed data for training.
        self.df_workout = self.create_dataframe(self.repo.get_all_workouts())
        self.df_intensity = self.create_dataframe(self.repo.get_intensities())
        self.df_mapping = self.create_dataframe(self.repo.get_mapping())

        # Do some data cleaning
        self.clean_mappings()

        # Create dataset # Store in vars, keep it flexable.
        self.data_x = self.create_lstm_data_x()
        self.data_y = self.create_lstm_data_y()

    @staticmethod
    def create_dataframe(db_data):
        data, cols = db_data
        return pd.DataFrame(data, columns=cols)

    def create_lstm_data_x(self):
        workouts = []
        for _, w in self.df_workout.groupby('w_id'):
            w = w.reindex(range(20), fill_value=0)
            w = w.drop(['w_id'], axis=1)
            w['e_sequence'] = range(1, 20+1)
            workouts.append(torch.tensor(w.values, dtype=torch.float32))
        return torch.stack(workouts)

    def create_lstm_data_y(self):
        y = torch.tensor(self.df_intensity['intensity'].values, dtype=torch.float32)
        y = y.unsqueeze(-1)
        return y

    def clean_mappings(self):
        self.df_mapping = pd.concat((pd.DataFrame({'mapid': [0], 'name': ['UNK']}), self.df_mapping), ignore_index=True)
        self.df_mapping = dict(zip(self.df_mapping['mapid'], self.df_mapping.index))
        self.df_workout['e_id'] = self.df_workout['e_id'].map(self.df_mapping).fillna(0).astype(int)


x = DNNTrainer()