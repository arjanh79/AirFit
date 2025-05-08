import torch
import torch.nn as nn
import numpy as np

class AirFitBiLSTM(nn.Module):
    def __init__(self):
        super().__init__()

        # LSTM params
        self.input_size = 6
        self.hidden_size = 10
        self.num_layers = 2

        self.num_exercises = 20 # 19 difference exercises... + 1 UNK
        self.embeddings_dim = 3 # Use a 3D representation per exercise

        self.embedding = nn.Embedding(self.num_exercises, self.embeddings_dim)
        self.features = nn.Linear(3, 3)
        with torch.no_grad():
            self.embedding.weight[0] = torch.zeros(self.embeddings_dim)
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.attn = nn.Linear(self.hidden_size * 2, 1)
        self.relu = nn.ReLU()


    def forward(self, e, f):
        e = self.embedding(e)
        f = f.reshape((-1, 20, 3)) # Reshape f, create a 3 vector per exercise
        mask = torch.tensor(np.where(f[:, :, 1] == 0, 0, 1))
        f = self.features(f)

        x = torch.cat((e, f), dim=2) # Output: torch.Size([batch, 20, 6])
        lstm_out, _ = self.lstm(x)

        attn_scores = self.attn(lstm_out).squeeze(-1)
        attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_scores = self.relu(attn_scores).unsqueeze(-1)

        intensity_per_exercise = self.relu(attn_scores * lstm_out)
        intensity_per_exercise = intensity_per_exercise.sum(dim=2)
        intensity_per_exercise = intensity_per_exercise * mask

        total_intensity = intensity_per_exercise.sum(dim=1).unsqueeze(-1)

        return total_intensity, intensity_per_exercise
