
import numpy as np

import torch
from torch.utils.data import Dataset


class IntensityDataset(Dataset):
    def __init__(self, workouts):
        self.workouts = workouts

        self.feature_cols = ['exercise_id', 'exercise_sequence', 'weight_id', 'reps',
                             'core', 'metric_type', 'equipment_id']

        self.y = workouts['workout_intensity'].astype('float32').to_numpy()

        self.x = []
        for row in self.workouts[self.feature_cols].itertuples(index=False, name=None):
            self.x.append(np.vstack(row))

        self.x = np.stack(self.x, axis=0).astype(np.int64)
        self.x = self.x.transpose((0, 2, 1))

        self.weighted_loss = self.create_weighted_loss()


    def __len__(self):
        return len(self.workouts)


    def __getitem__(self, idx):
        x = torch.from_numpy(self.x[idx]).long()
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        l = self.weighted_loss[idx]

        return x, y, l


    def create_weighted_loss(self):
        factor = 0.975
        weighted_loss = [factor ** i for i in range(len(self.x))]
        weighted_loss = torch.tensor(weighted_loss, dtype=torch.float32).unsqueeze(1)
        weighted_loss = torch.clip(weighted_loss, 0.05, 1)
        return weighted_loss