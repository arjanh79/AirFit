import random

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F



class WorkoutGeneratorRNN:
    def __init__(self):
        self.workouts, self.exercises, self.weights = self.get_workouts()
        self.ex_to_id, self.id_to_ex = self.get_tokens()

    @staticmethod
    def get_workouts():
        with open('trainingblocks', 'r') as f:
            data = f.readlines()

        data = [i.replace('\n', '').split('\t') for i in data]
        df = pd.DataFrame(data, columns=['block_id', 'seq', 'description', 'exercise_id', 'name'])
        df = df[['block_id', 'seq', 'name']]
        df = df.sort_values(['block_id', 'seq'])


        exercise_count = df['name'].value_counts().reset_index()
        total_count = exercise_count['count'].sum()
        exercise_count['rel_count'] = exercise_count['count'] / total_count
        exercise_count = exercise_count[['name', 'rel_count']]
        exercise_count['rel_count'] = 1 - exercise_count['rel_count']

        workouts = df.groupby('block_id')['name'].apply(list)
        exercises = df['name'].drop_duplicates().sort_values().to_list()

        return list(workouts), exercises, exercise_count

    def get_tokens(self):
        ex_to_id = {v: c+1 for c, v in enumerate(self.exercises)}
        id_to_ex = {c+1: v for c, v in enumerate(self.exercises)}
        return ex_to_id, id_to_ex

    def tokenize(self):
        return [
            [self.ex_to_id[ex] for ex in workout]
            for workout in self.workouts
        ]

    def get_max_token_id(self):
        return max(self.ex_to_id.values())

class WorkoutDataset(Dataset):
    def __init__(self, workouts, window_size=2):
        self.window_size = window_size
        self.samples = [(workout[i:i + self.window_size], workout[i + self.window_size]) for workout in workouts for i in range(len(workout) - window_size)]
        print(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        window, target = self.samples[idx]
        window = torch.tensor(window, dtype=torch.long)
        target = torch.tensor(target, dtype=torch.long)
        return window, target

class MyModel(nn.Module):
    def __init__(self, num_embeddings):
        super().__init__()
        self.embeddings_dim = 4
        self.num_embeddings = int(num_embeddings)
        self.hidden_size = self.embeddings_dim * 4
        self.num_layers = 1
        self.drop = nn.Dropout(p=0.2)

        self.embeddings = nn.Embedding(
            num_embeddings=self.num_embeddings,
            embedding_dim=self.embeddings_dim,
            padding_idx=0,
            max_norm=3
        )

        self.gru = nn.GRU(input_size=self.embeddings_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
            )

        self.fc = nn.Linear(self.hidden_size, self.num_embeddings, bias=False)

    def forward(self, x):
        x = self.embeddings(x)
        x = self.drop(x)
        x, _ = self.gru(x)
        x = x[:, -1, :]
        x = self.drop(x)
        x = self.fc(x)
        return x

class Trainer:
    def __init__(self):
        self.wg = WorkoutGeneratorRNN()
        self.ds = WorkoutDataset(self.wg.tokenize())
        self.dl = DataLoader(self.ds, batch_size=16, shuffle=True)

        self.num_embeddings = self.wg.get_max_token_id() + 1

        self.model = MyModel(self.num_embeddings)

        self.epochs = 5000
        self.optimizer = torch.optim.NAdam(self.model.parameters(), lr=0.001)
        self.loss = nn.CrossEntropyLoss()

    def train(self):
        best_eval_loss = float('inf')
        no_improvement = 0
        for epoch in range(self.epochs):
            self.model.train()
            for batch, (x, y) in enumerate(self.dl):
                self.optimizer.zero_grad()
                y_pred = self.model(x)
                loss = self.loss(y_pred, y)
                loss.backward()
                self.optimizer.step()
                print(f'Epoch: {epoch+1}, batch: {batch+1}, loss: {loss.item():.5f}, ({len(y)})')
            eval_loss = self.eval()
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
            else:
                no_improvement += 1
                if no_improvement >= 5:
                    break

    def eval(self):
        self.model.eval()
        total_loss = 0
        samples = 0
        with torch.no_grad():
            for batch, (x, y) in enumerate(self.dl):
                y_pred = self.model(x)
                loss = self.loss(y_pred, y)
                total_loss += (loss.item() * len(y))
                samples += len(y)
        eval_loss = total_loss / samples
        print(f'               EVAL LOSS: {eval_loss:.5f}\n')
        return eval_loss

    def create_workout(self, length=6):
        tokens = self.wg.tokenize()
        base_workout = random.choice(tokens)
        temperature = 0.85

        self.wg.weights['token_id'] = self.wg.weights['name'].apply(lambda x: self.wg.ex_to_id[x])
        training_weights = self.wg.weights.sort_values(['token_id'])['rel_count'].to_list()
        training_weights.insert(0, 0)

        training_weights = torch.Tensor(training_weights).unsqueeze(0) * 2

        workout = base_workout[:2]

        for _ in range(length - 2):
            x = torch.tensor(workout[-2:], dtype=torch.long).unsqueeze(0)
            logits = self.model(x)

            logits = logits * training_weights

            logits = logits.clone()
            block = set(workout) | {0}
            logits[0, list(block)] = -1e9


            logits /= temperature
            logits = F.softmax(logits, dim=1)

            selected_exercise = torch.multinomial(logits, 1).item()
            workout.append(selected_exercise)

        for i in workout:
            print(self.wg.id_to_ex[i])



t = Trainer()
# t.train()
# for _ in range(5):
#    t.create_workout(length=12)
#    print('-'*10)
