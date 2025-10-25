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

        self.features_1 = nn.Linear(5, 5)
        self.features_2 = nn.Linear(5, 3)

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
        self.selu = nn.SELU()
        self.dropout = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(5)


    def forward(self, e, f):

        B, T = e.shape[:2]   # Batch, Training length

        e = self.embedding(e)
        e = self.selu(e)

        f = f.view(B, T, -1)

        lengths = torch.count_nonzero(f[:, :, 1], dim=1).clamp_min(1)  # Find the true length of the workout.
        f0_sq = f[:, :, 0:1] ** 2
        f0_sq = torch.log1p(f0_sq)

        # Create embeddings for the sequence numbers.
        seq_idx = f[:, :, -1].to(torch.long)
        seq_emb = self.seq_embedding(seq_idx)

        # Merge weights, reps and seq_nums
        f_cat = torch.cat((f0_sq, f[:, :, 1:-1], seq_emb), dim=2)
        f_cat = self.norm(f_cat)

        # preprocess the output of the previous step
        f_feat = self.features_1(f_cat)
        f_feat = self.selu(f_feat)
        f_feat = self.features_2(f_feat)
        f_feat = self.selu(f_feat)

        # Merge the output of the feature with the embeddings of the exercise.
        x = torch.cat((e, f_feat), dim=2)

        # Process the workouts in an LSTM with variable workout lengths.

        # If all workouts in the batch have the same length, nothing to worry about.
        if torch.all(lengths == T):
            lstm_out, _ = self.lstm(x)
        # If not, we use PackedSequence to handle the difference in lengths
        else:
            packed = torch.nn.utils.rnn.pack_padded_sequence(
                x, lengths=lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.lstm(packed)
            lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(
                packed_out, batch_first=True, total_length=T
            )

        # Now we have overfitting, adding some dropout

        lstm_out = self.dropout(lstm_out)

        # This works magic, init at .5 without bias. Sum is to get the total workout intensity.
        intensity_per_exercise = self.intensity_head(lstm_out).squeeze(-1)
        total_intensity = intensity_per_exercise.sum(dim=1, keepdim=True)

        return total_intensity, intensity_per_exercise
