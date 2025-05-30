import torch
import torch.nn as nn
import torch.optim as optim


class ModelTraining:
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset

        self.model_location = '../AI/workout_model.pth'
        self.model_location_train = '../AI/workout_model_train.pth'

        self.epochs = 5  # 5
        self.batch_size = 32
        self.lr = 0.005
        self.safe_model = False # FALSE!!
        self.load_model = True # FALSE!!

        if self.load_model:
            self.model.load_state_dict(torch.load(self.model_location))

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
                output, _ = self.model(Xe, Xf)
                loss = loss_fn(output, y)
                loss = torch.mean(loss * wl)
                loss.backward()
                optimizer.step()
            print(f'Epoch {epoch+1:03d}, Loss: {loss.item():>8.5f}')
        if self.safe_model:
            torch.save(self.model.state_dict(), self.model_location)
        else:
            torch.save(self.model.state_dict(), self.model_location_train)


    def eval_model(self, test_model=False):
        if self.load_model:
            self.model.load_state_dict(torch.load(self.model_location))
        if test_model and self.load_model:
            print('\nLoading training model...')
            self.model.load_state_dict(torch.load(self.model_location_train))

        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=32, shuffle=False)
        self.model.eval()
        with torch.no_grad():
            for Xe, Xf, y, _ in dataloader:
                y_hat, _ = self.model(Xe, Xf)
                print('---- Model performance:')
                print(f'y_true: {y.flatten()}')
                print(f'y_hat: {y_hat.flatten()}')


    def eval_workout(self):
        self.model.load_state_dict(torch.load(self.model_location))
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=1, shuffle=False)

        if not len(dataloader) == 1:
            return -1

        self.model.eval()
        with torch.no_grad():
            Xe, Xf, _, _ = next(iter(dataloader))
            y_hat = self.model(Xe, Xf)
        return y_hat