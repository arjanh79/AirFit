from v3.data.get_workouts import Workouts

from torch.utils.data import TensorDataset, DataLoader

if __name__ == '__main__':
    w = Workouts()
    w.generate()
    x, y = w.to_tensor()

    train_dataset = TensorDataset(x, y)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)

    for x, y in train_loader:
        print(x)
        print(y)

