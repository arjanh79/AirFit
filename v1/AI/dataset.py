from torch.utils.data import Dataset
import numpy as np

class WorkoutDataset(Dataset):
    def __init__(self, Xe, Xf, y, wl):
        self.Xe = Xe
        self.Xf = Xf
        self.y = y
        self.wl = wl
        self.wo_length = self.get_workout_length()

    def get_workout_length(self):
        mask = self.Xe < 1
        idx = np.argmax(mask, axis=1)
        has_match = mask.any(axis=1)
        idx[~has_match] = 19
        idx += 1
        return idx

    def __len__(self):
        return len(self.Xe)


    def __getitem__(self, idx):
        return self.Xe[idx], self.Xf[idx], self.y[idx], self.wl[idx], self.wo_length[idx]


    def __repr__(self):
        summary = (f'WorkoutDataset('
                f'samples={len(self)}, '
                f'embeddings_shape={self.Xe.shape}, '
                f'data_x_shape={self.Xf.shape}, '
                f'data_y_shape={self.y.shape}, '
                f'weighted_loss_shape={self.wl.shape})'
                )
        return summary