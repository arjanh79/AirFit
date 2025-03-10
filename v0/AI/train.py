import torch
import torch.nn as nn
import torch.optim as optim

def get_loss():
    return nn.MSELoss(reduction='none')

def get_optimizer(model, lr):
    return optim.NAdam(model.parameters(), lr=lr)


def train_model(model, dataset, epochs=8, batch_size=32, lr=0.003):

    model.load_state_dict(torch.load("../workout_model.pth"))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    loss_fn = get_loss()
    optimizer = get_optimizer(model, lr)

    model.train()

    for epoch in range(epochs):
        for Xe, Xf, y, wl in dataloader:
            optimizer.zero_grad()
            output = model(Xe, Xf)
            loss = loss_fn(output, y)
            loss = torch.mean(loss * wl)
            loss.backward()
            optimizer.step()
    print(output.flatten(), loss.item())
    # torch.save(model.state_dict(), '../workout_model.pth')

def eval_model(model, dataset):
    model.load_state_dict(torch.load("../workout_model.pth"))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
    model.eval()
    with torch.no_grad():
        for Xe, Xf, y, _ in dataloader:
            pred = model(Xe, Xf)
            print(f'PRED: {pred.flatten()}')
            print(f'EXP: {y.flatten()}')
