import torch
import torch.nn as nn
import torch.optim as optim

def get_loss():
    return nn.MSELoss()

def get_optimizer(model, lr):
    return optim.NAdam(model.parameters(), lr=lr)


def train_model(model, dataset, epochs=10, batch_size=32, lr=0.005):

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    loss_fn = get_loss()
    optimizer = get_optimizer(model, lr)

    model.train()

    for epoch in range(epochs):
        for Xe, Xf, y in dataloader:
            optimizer.zero_grad()
            output = model(Xe, Xf)
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()
            print(output.item(), loss.item())
    # torch.save(model.state_dict(), '../workout_model.pth')