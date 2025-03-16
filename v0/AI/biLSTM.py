import torch
import torch.nn as nn
import numpy as np

class AirFitBiLSTM(nn.Module):
    def __init__(self):
        super().__init__()

        self.num_exercises = 13 # 13 difference exercises
        self.embeddings_dim = 3 # Use a 3D representation per exercise

        self.embedding = nn.Embedding(self.num_exercises, self.embeddings_dim)
        self.features = nn.Linear(3, 3)
        with torch.no_grad():
            self.embedding.weight[0] = torch.zeros(self.embeddings_dim)
        self.lstm = nn.LSTM(
            input_size=6, hidden_size=10, num_layers=2, batch_first=True, bidirectional=True
        )
        self.relu = nn.ReLU()


    def forward(self, e, f):
        e = self.embedding(e)
        f = f.reshape((-1, 20, 3)) # Reshape f, create a 3 vector per exercise
        mask = torch.tensor(np.where(f[:, :, 1] == 0, 0, 1))
        f = self.features(f)

        x = torch.cat((e, f), dim=2) # Output: torch.Size([batch, 20, 6])
        lstm_out, _ = self.lstm(x)
        intensity_per_exercise = lstm_out[:, :, -1]
        intensity_per_exercise *= mask
        intensity_per_exercise = self.relu(intensity_per_exercise)

        total_intensity = intensity_per_exercise.sum(dim=1).unsqueeze(-1)

        return total_intensity, intensity_per_exercise
