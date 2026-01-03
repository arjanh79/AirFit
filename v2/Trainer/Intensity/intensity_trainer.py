import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from v2.Data.intensity_dataset import IntensityDataset
from v2.Domain.intensity_combinator import IntensityCombinator
from v2.Trainer.Intensity.intensity_model import IntensityTransformer
from v2.config import MODEL_PATH


class IntensityTrainer:

    def __init__(self, combinator: IntensityCombinator, dataset: IntensityDataset,
                 num_embeddings: dict[str, int], col_names: list[str]):

        self.wc = combinator
        self.ds = dataset
        self.dl = DataLoader(self.ds, batch_size=16, shuffle=True)
        self.num_embeddings = num_embeddings
        self.col_names = col_names

        self.model = IntensityTransformer(num_embeddings=self.num_embeddings, col_names=self.col_names)

        self.optimizer = optim.NAdam(self.model.parameters(), lr=1e-3)
        self.loss_fn = nn.MSELoss()
        self.scheduler = self.get_scheduler()


    def train_one_epoch(self, epoch_num) -> float:
        self.model.train()
        total_loss = 0.0
        for batch, (x, y) in enumerate(self.dl):
            self.optimizer.zero_grad()
            intensity = self.model(x)  # (B, 1)

            loss = self.loss_fn(intensity, y.unsqueeze(1))

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            print(f'    [TRN] Epoch: {epoch_num} - Batch: {batch}, Batch loss: {loss.item():.5f}')

        loss = total_loss / len(self.dl)
        return loss

    def eval(self):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for x, y in self.dl:
                intensity = self.model(x)
                loss = self.loss_fn(intensity, y.unsqueeze(1))
                total_loss += loss.item()
        eval_loss = total_loss / len(self.dl)
        self.scheduler.step(eval_loss)

        return eval_loss


    def fit(self, epochs: int) -> None:

        epochs_without_improve = 0
        best_eval_loss = float('inf')
        best_epoch = 0
        patience = 20

        for epoch in range(1, epochs+1):
            _  = self.train_one_epoch(epoch)  # Output not yet needed
            eval_loss = self.eval()
            print(f' >> [EVL] Epoch: {epoch} - Loss: {eval_loss:.5f}')

            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                epochs_without_improve = 0
                best_epoch = epoch
                self.save_model('best')
            else:
                epochs_without_improve += 1

            print('-' * 53)

            if epochs_without_improve >= patience:
                print('\n'+'=' * 67)
                print(f' *** Early stopping at Epoch {epoch}. Best loss: {best_eval_loss:.3f} @ Epoch {best_epoch} ***')
                print('=' * 67)
                break



    def get_scheduler(self) -> ReduceLROnPlateau:
        scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.75,
            patience=4,
            threshold_mode='rel',
            threshold=1e-3,
            min_lr=1e-6,
            cooldown=3,
        )
        return scheduler


    def save_model(self, tag) -> None:
        torch.save(self.model.state_dict(), MODEL_PATH / f'intensity_model_{tag}.pth')
