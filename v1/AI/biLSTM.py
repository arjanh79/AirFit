import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


class AirFitBiLSTM(nn.Module):
    def __init__(self):
        super().__init__()

        # LSTM params
        self.input_size = 8
        self.hidden_size = 10
        self.num_layers = 5

        self.num_exercises = 24 # 23 Different exercises... + 1 UNK
        self.embeddings_dim = 5 # Use a 5D representation per exercise
        self.embedding = nn.Embedding(self.num_exercises, self.embeddings_dim, max_norm=3.0)

        self.seq_len = 21
        self.seq_dim = 3
        self.seq_embedding = nn.Embedding(self.seq_len, self.seq_dim, max_norm=3.0)

        self.features = nn.Linear(5, 3)

        self.intensity_head = nn.Linear(self.hidden_size * 2, 1, bias=False)
        torch.nn.init.normal_(self.intensity_head.weight, mean=0.5, std=0.05)

        with torch.no_grad():
            self.embedding.weight[0] = torch.zeros(self.embeddings_dim)

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )

        self.relu = nn.ReLU()


    def forward(self, e, f):

        B, T = e.shape[:2]

        e = self.embedding(e)
        f = f.view(B, T, -1)

        lengths = torch.count_nonzero(f[:, :, 1], dim=1).clamp_min(1)
        f0_sq = f[:, :, 0:1] * f[:, :, 0:1]

        seq_idx = f[:, :, -1].to(torch.long)
        seq_emb = self.seq_embedding(seq_idx)

        f_cat = torch.cat((f0_sq, f[:, :, 1:-1], seq_emb), dim=2)

        f_feat = self.features(f_cat)


        x = torch.cat((e, f_feat), dim=2)

        if torch.all(lengths == T):
            lstm_out, _ = self.lstm(x)
        else:
            packed = torch.nn.utils.rnn.pack_padded_sequence(
                x, lengths=lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.lstm(packed)
            lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(
                packed_out, batch_first=True, total_length=T
            )

        intensity_per_exercise = self.intensity_head(lstm_out).squeeze(-1)
        total_intensity = intensity_per_exercise.sum(dim=1, keepdim=True)

        return total_intensity, intensity_per_exercise
