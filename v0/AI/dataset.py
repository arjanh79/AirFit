from torch.utils.data import Dataset


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