import torch

from torch.utils.data import Dataset


class WorkoutDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.data_file = 'data/workouts.txt'
        self.X, self.y = self.read_data()
        self.weighted_loss = self.create_weighted_loss()

        print(self.X)
        print(self.y)
        print(self.weighted_loss)


    def read_data(self):
        with open(self.data_file, 'r') as f:
            lines = [line.strip().split(',') for line in f]

        lines = [[int(x) for x in line] for line in lines]
        lines = torch.tensor(lines, dtype=torch.long)

        return lines[:, :6].float().flip(0), lines[:, 6:].float().flip(0)


    def create_weighted_loss(self):

        n = self.X.shape[0]
        cutoff = min(n, 30)

        x = torch.linspace(0, torch.pi, steps=cutoff)
        x = torch.cos(x) * 0.475 + 0.525
        x = torch.clamp(x, 0.05, 1)

        if n > cutoff:
            tail = torch.full((n - cutoff,), 0.05)
            x = torch.cat((x, tail))

        return x


    def __len__(self):
        return self.X.shape[0]


    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.weighted_loss[idx]


if __name__ == '__main__':
    wo = WorkoutDataset()
