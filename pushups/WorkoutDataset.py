import torch

from torch.utils.data import Dataset


class WorkoutDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.data_file = 'data/workouts.txt'
        self.X, self.y = self.read_data()


    def read_data(self):
        with open(self.data_file, 'r') as f:
            lines = [line.strip().split(',') for line in f]

        lines = [[int(x) for x in line] for line in lines]
        lines = torch.tensor(lines, dtype=torch.long)

        return lines[:, :6].float(), lines[:, 6:].float()


    def __len__(self):
        return self.X.shape[0]


    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


if __name__ == '__main__':
    wo = WorkoutDataset()
