import torch
import torch.nn as nn
import numpy as np

class AirFitBiLSTM(nn.Module):
    def __init__(self):
        super().__init__()

        # LSTM params
        self.input_size = 8
        self.hidden_size = 10
        self.num_layers = 2

        self.num_exercises = 24 # 23 Different exercises... + 1 UNK
        self.embeddings_dim = 5 # Use a 3D representation per exercise
        self.embedding = nn.Embedding(self.num_exercises, self.embeddings_dim, max_norm=3.0)

        self.seq_len = 21
        self.seq_dim = 2
        self.seq_embedding = nn.Embedding(self.seq_len, self.seq_dim, max_norm=3.0)

        self.features = nn.Linear(4, 3)

        with torch.no_grad():
            self.embedding.weight[0] = torch.zeros(self.embeddings_dim)

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True
        )

        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(3)


    def forward(self, e, f):
        e = self.embedding(e)
        f = f.reshape((-1, 20, 3)) # Reshape f, create a 3 vector per exercise
        mask = torch.tensor(np.where(f[:, :, 1] == 0, 0, 1))

        seq_nums = self.seq_embedding(f[:, :, -1].int())
        seq_nums = seq_nums.squeeze(-1)
        f = torch.cat((f[:, :, :-1], seq_nums), dim=2)

        f = self.features(f)
        f = self.layer_norm(f)

        x = torch.cat((e, f), dim=2) # Output: torch.Size([batch, 20, 6])

        lstm_out, _ = self.lstm(x)

        intensity_per_exercise = self.relu(lstm_out)
        intensity_per_exercise = intensity_per_exercise.sum(dim=2)
        intensity_per_exercise = intensity_per_exercise * mask

        total_intensity = intensity_per_exercise.sum(dim=1).unsqueeze(-1)

        return total_intensity, intensity_per_exercise
