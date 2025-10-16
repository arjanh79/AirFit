
import torch

import pandas as pd
import numpy as np

from v1.AI.dataset import WorkoutDataset
from v1.DB.factories import RepositoryFactory
from v1.utils.tools import get_loss_decay


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

        # Create dataset -  Store in vars, keep it flexable.
        self.data_x = self.create_lstm_data_x()
        self.data_y = self.create_lstm_data_y()

        # Reshape the data for a DNN
        self.embeddings_x, self.data_x = self.create_dnn_data_x()

        # Calculate weight factor for loss
        self.weighted_loss = get_loss_decay(self.data_x.shape[0])

        self.ds = WorkoutDataset(self.embeddings_x, self.data_x, self.data_y, self.weighted_loss)


    @staticmethod
    def create_dataframe(db_data):
        data, cols = db_data
        return pd.DataFrame(data, columns=cols)


    def create_dnn_data_x(self):
        embeddings = self.data_x[:, :, 0].int()
        self.data_x = torch.cat((self.data_x[:, :, 1:], ), dim=2)
        self.data_x = self.data_x.flatten(start_dim=1, end_dim=2)
        return embeddings, self.data_x


    def create_lstm_data_x(self):
        workouts = []
        for _, w in self.df_workout.groupby('w_id', sort=False):
            w = w.reset_index(drop=True)
            w = w.reindex(range(20), fill_value=0)
            w = w.drop(['w_id'], axis=1)
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
