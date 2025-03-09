import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

import pandas as pd

from dev.v0.AI.AirFitDNN import AirFitDNN
from dev.v0.DB.factories import RepositoryFactory

class WorkoutDataset(Dataset):
    def __init__(self, Xe, Xf, y):
        self.Xe = Xe
        self.Xf = Xf
        self.y = y

    def __len__(self):
        return len(self.Xe)

    def __getitem__(self, idx):
        return self.Xe[idx], self.Xf[idx], self.y[idx]

    def __repr__(self):
        summary = (f'WorkoutDataset('
                f'samples={len(self)}, '
                f'embeddings_shape={self.Xe.shape}, '
                f'data_x_shape={self.Xf.shape}, '
                f'data_y_shape={self.y.shape})')
        return summary



class WorkoutPreprocessor:
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

        # Reshape the data for a DNN
        self.embeddings_x, self.data_x = self.create_dnn_data_x()

        self.ds = WorkoutDataset(self.embeddings_x, self.data_x, self.data_y)

    @staticmethod
    def create_dataframe(db_data):
        data, cols = db_data
        return pd.DataFrame(data, columns=cols)

    def create_dnn_data_x(self):
        embeddings = self.data_x[:, :, 0]
        self.data_x = torch.cat((self.data_x[:, :, 1:], ), dim=2)
        self.data_x = self.data_x.flatten(start_dim=1, end_dim=2)
        return embeddings, self.data_x

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

    def get_dataset(self):
        return self.ds


class WorkoutTrainer:
    def __init__(self):
        # Get the data
        self.ds = WorkoutPreprocessor().get_dataset()
        self.dl = DataLoader(self.ds)

        # Load the model
        self.model = AirFitDNN()

    def train(self):
        for epoch in range(10):
            self.model.train()
            for batch, (Xe, Xf, y) in enumerate(self.dl):
                y_hat = self.model(Xe, Xf)
                loss = self.model.loss_fn(y_hat, y)
                loss.backward()
                self.model.optimizer.step()
                self.model.optimizer.zero_grad()

wt = WorkoutTrainer()
wt.train()