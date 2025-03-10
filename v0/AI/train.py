import torch
import torch.nn as nn
import torch.optim as optim

class ModelTraining:
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset

        self.model_location = '../workout_model.pth'
        self.model.load_state_dict(torch.load(self.model_location))

        self.epochs = 3
        self.batch_size = 32
        self.lr = 0.003
        self.safe_model = False

        self.loss = self.get_loss()
        self.optimizer = self.get_optimizer()

    @staticmethod
    def get_loss():
        return nn.MSELoss(reduction='none')


    def get_optimizer(self):
        return optim.NAdam(self.model.parameters(), lr=self.lr)


    def train_model(self):
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        loss_fn = self.get_loss()
        optimizer = self.get_optimizer()
        self.model.train()

        for epoch in range(self.epochs):
            for Xe, Xf, y, wl in dataloader:
                optimizer.zero_grad()
                output = self.model(Xe, Xf)
                loss = loss_fn(output, y)
                loss = torch.mean(loss * wl)
                loss.backward()
                optimizer.step()
        self.eval_model()
        if self.safe_model:
            torch.save(self.model.state_dict(), self.model_location)

    def eval_model(self):
        self.model.load_state_dict(torch.load(self.model_location))

        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        self.model.eval()
        with torch.no_grad():
            for Xe, Xf, y, _ in dataloader:
                y_hat = self.model(Xe, Xf)
                print('---- Model performance:')
                print(f'y_hat: {y_hat.flatten()}')
                print(f'y_true: {y.flatten()}')