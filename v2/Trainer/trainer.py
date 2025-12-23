import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from v2.Data.workout_dataset import WorkoutDataset
from v2.Domain.workout_combinator import WorkoutCombinator
from v2.Trainer.workout_model import WorkoutTransformer
from v2.config import MODEL_PATH


class WorkoutTrainer:
    def __init__(self, combinator: WorkoutCombinator, dataset: WorkoutDataset):

        self.wc = combinator
        self.ds = dataset
        self.dl = DataLoader(self.ds, batch_size=16, shuffle=True)

        self.model_params = self.get_model_parameters()

        self.model = WorkoutTransformer(self.model_params['vocab_size'],
                                        self.model_params['d_model'],
                                        self.model_params['n_head'],
                                        self.model_params['num_layers'],
                                        self.model_params['max_len'])

        self.optimizer = optim.NAdam(self.model.parameters(), lr=1e-3)
        self.loss_fn = nn.CrossEntropyLoss()


    def get_model_parameters(self) -> dict[str, int]:
        model_params = dict()
        model_params['vocab_size'] = len(self.ds.ex_to_id) + 2
        model_params['d_model'] = 8
        model_params['n_head'] = 2
        model_params['num_layers'] = 2
        model_params['max_len'] = 13
        return model_params

    def train_one_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        for batch, (x, y) in enumerate(self.dl):
            self.optimizer.zero_grad()
            logits = self.model(x)  # (B, T, vocab)

            loss = self.loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            print(f'Batch: {batch}, loss: {loss.item():.5f}')

        loss = total_loss / max(1, len(self.dl))
        return loss


    def fit(self, epochs: int=100, patience:int=10) -> None:
        epochs_without_improve = 0
        best_eval_loss = float('inf')

        for epoch in range(1, epochs+1):
            print(f'Epoch {epoch+1}')

            self.train_one_epoch()
            eval_loss = self.eval()
            print(f' Eval loss: {eval_loss:.5f}')

            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                epochs_without_improve = 0
                self.save_model("best")
                print(' New best model saved')
            else:
                epochs_without_improve += 1

            if epochs_without_improve >= patience:
                print(f'\n*** Early stopping at epoch {epoch+1} ***')
                break

            print('-'*24)


    def eval(self):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for x, y in self.dl:
                logits = self.model(x)
                loss = self.loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                total_loss += loss.item()
        return total_loss / len(self.dl)


    def save_model(self, tag) -> None:
        torch.save(self.model.state_dict(), MODEL_PATH / f'airfit_model_{tag}.pth')