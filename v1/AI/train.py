import torch
import torch.nn as nn
import torch.optim as optim

import pandas as pd

class ModelTraining:
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset

        self.model_location = 'AI/workout_model.pth'
        self.model_location_train = 'AI/workout_model_train.pth'

        self.epochs = 5  # 5
        self.batch_size = self.calc_batch_size()
        self.lr = 0.003 # 0.005
        self.safe_model = False # FALSE!!
        self.load_model = True # FALSE!!

        if self.load_model:
            self.model.load_state_dict(torch.load(self.model_location))

        self.loss = self.get_loss()
        self.optimizer = self.get_optimizer()


    def calc_batch_size(self):
        total_samples = len(self.dataset)
        estimate = int(total_samples ** 0.5) + 1

        last_batch = total_samples % estimate
        min_last_batch = estimate - 1

        while 0 < last_batch < min_last_batch:
            estimate += 1
            last_batch = total_samples % estimate

        return estimate


    @staticmethod
    def get_loss():
        return nn.MSELoss(reduction='none')

    def param_groups_decay_only_embeddings(self, wd=1e-4):
        emb_params, other_params = [], []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if n.endswith("embedding.weight") or "embedding.weight" in n:
                emb_params.append(p)
            else:
                other_params.append(p)
        return [
            {"params": emb_params, "weight_decay": wd},
            {"params": other_params, "weight_decay": 0.0},
        ]

    def get_optimizer(self):
        # return optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.03)
        groups = self.param_groups_decay_only_embeddings(wd=1e-3)
        return torch.optim.AdamW(groups, lr=3e-3)


    def train_model(self):
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        optimizer = self.get_optimizer()
        self.model.train()
        for epoch in range(self.epochs):
            for batch, (Xe, Xf, y, wl) in enumerate(dataloader, 1):
                optimizer.zero_grad()
                output, _ = self.model(Xe, Xf)
                loss = self.calculate_loss(output, y, wl)
                loss = torch.mean(loss)
                loss.backward()
                optimizer.step()
                print(f'   Epoch {epoch + 1:03d}, Batch {batch:03d}: {loss.item():>8.5f} ({len(y)})')
            epoch_loss = self.calculate_epoch_loss()
            print(f'=> Epoch {epoch + 1:03d}       Loss: {epoch_loss:>8.5f}\n')
        if self.safe_model:
            torch.save(self.model.state_dict(), self.model_location)
        torch.save(self.model.state_dict(), self.model_location_train)


    def calculate_loss(self, y_hat, y_true, wl):
        loss = self.loss(y_hat, y_true) * wl
        return loss


    def calculate_epoch_loss(self):
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=1, shuffle=False)
        self.model.eval()
        epoch_loss = 0
        with torch.no_grad():
            for Xe, Xf, y, wl in dataloader:
                y_hat, _ = self.model(Xe, Xf)
                loss = self.loss(y_hat, y) * wl
                epoch_loss += loss.item()
        epoch_loss = epoch_loss / len(dataloader)
        return epoch_loss


    def eval_model(self):
        self.eval_model_helper(prod=False)
        self.eval_model_helper(prod=True)


    def eval_model_helper(self, prod):
        if prod:
            print('> Model on disk:')
            self.model.load_state_dict(torch.load(self.model_location))
        else:
            print('> Last trained model:')
            self.model.load_state_dict(torch.load(self.model_location_train))
        loss = self.calculate_epoch_loss()
        print(f'Loss: {loss:>8.5f}')


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


    def make_pred(self):
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=1, shuffle=False)

        self.model.load_state_dict(torch.load(self.model_location_train))
        self.model.eval()

        result = []
        with torch.no_grad():
            for Xe, Xf, y, wl in dataloader:
                y_hat, _ = self.model(Xe, Xf)
                result.append((y.item(), round(y_hat.item(), 3), round(wl.item(), 3)))

        df = pd.DataFrame(result, columns=['True', 'Pred', 'WL'])
        df['Gain'] = df['True'] - df['Pred']
        print(df)
