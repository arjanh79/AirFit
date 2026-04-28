import torch
import numpy as np

from pushups.WorkoutModel import WorkoutModel

import matplotlib.pyplot as plt


class WorkoutGenerate:
    def __init__(self):
        self.model = WorkoutModel()
        self.model.load_state_dict(torch.load('best_model.pth'))


    def evaluate(self):
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        reps = torch.nn.Parameter(torch.normal(4, 2, (1,6)))
        target = torch.full((1, 6), 0.85)

        optimizer = torch.optim.NAdam([reps], lr=0.1)
        loss_fn = torch.nn.MSELoss()

        previous_loss = torch.inf
        for step in range(100):
            optimizer.zero_grad()
            logits = self.model(reps)
            probs = torch.sigmoid(logits)
            loss =loss_fn(probs, target)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                self.update_reps(reps)
            print(np.round(reps.detach().numpy(), 3))
            diff_loss = abs(previous_loss - loss.item())
            if diff_loss < 1.0e-05 and step >= 10:
                print(f'{step} {diff_loss:.5f}')
                break
            previous_loss = loss.item()

        with torch.no_grad():
            self.update_reps(reps)

        reps = np.round(reps.squeeze().detach().numpy())
        reps[[2, 4]] = reps[[2, 4]] + reps[2] % 2
        print(reps.astype(int))




    def update_reps(self, reps):
        push_ups = torch.mean(reps[:, [1, 3, 5]])
        step_ups = torch.mean(reps[:, [2, 4]])

        reps[:, [1, 3, 5]] = push_ups
        reps[:, [2, 4]] = step_ups


if __name__ == '__main__':
    workout = WorkoutGenerate()
    workout.evaluate()
