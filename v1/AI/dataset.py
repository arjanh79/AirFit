from torch.utils.data import Dataset

class WorkoutDataset(Dataset):
    def __init__(self, Xe, Xf, y, wl):
        self.Xe = Xe.flip(dims=(0, ))
        self.Xf = Xf.flip(dims=(0, ))
        self.y = y.flip(dims=(0, ))
        self.wl = wl.flip(dims=(0, ))

    def __len__(self):
        return len(self.Xe)


    def __getitem__(self, idx):
        return self.Xe[idx], self.Xf[idx], self.y[idx], self.wl[idx]


    def __repr__(self):
        summary = (f'WorkoutDataset('
                f'samples={len(self)}, '
                f'embeddings_shape={self.Xe.shape}, '
                f'data_x_shape={self.Xf.shape}, '
                f'data_y_shape={self.y.shape}, '
                f'weighted_loss_shape={self.wl.shape})'
                )
        return summary