
import torch
import torch.nn as nn
import torch.optim as optim


from torch.utils.data import DataLoader
from pushups.WorkoutDataset import WorkoutDataset
from pushups.WorkoutModel import WorkoutModel

if __name__ == '__main__':
    model = WorkoutModel()

    model.load_state_dict(torch.load('best_model.pth'))

    ds = WorkoutDataset()
    dl = DataLoader(ds, batch_size=32, shuffle=False)

    loss_fn = nn.BCEWithLogitsLoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=0.001)


    best_loss = torch.inf
    loss_counter = 0
    for epoch in range(500):
        model.train()
        for X, y, l in dl:
            model.zero_grad()
            y_pred = model(X)

            loss = loss_fn(y_pred, y)
            loss = loss.sum(dim=1)
            loss = (loss * l).mean()
            loss.backward()

            optimizer.step()
            print(f'Epoch: {epoch:03d} Loss: {loss.item():.5f}')
        model.eval()
        with torch.no_grad():
            total_loss = 0
            for X, y, l in dl:
                y_pred = model(X)

                loss = loss_fn(y_pred, y)
                loss = loss.sum(dim=1)
                loss = (loss * l).mean()

                total_loss += loss
            if total_loss < best_loss:
                best_loss = total_loss
                print(f'>> Saving best model: {epoch:03d} {best_loss.item():.5f}')
                torch.save(model.state_dict(), 'best_model.pth')
                loss_counter = 0
            else:
                loss_counter += 1
                if loss_counter == 10:
                    print('EARLY STOPPING!')
                    break