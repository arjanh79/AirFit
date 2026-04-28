import torch
import torch.nn as nn


class WorkoutModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.exercise_ids = torch.tensor([1, 2, 3, 2, 3, 2], dtype=torch.long)
        self.seq_ids = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.long)

        self.embedding_dim = 2

        self.exercise_embedding = nn.Embedding(4, self.embedding_dim, padding_idx=0)
        self.seq_embedding = nn.Embedding(7, self.embedding_dim, padding_idx=0)

        self.input_projection = nn.Linear(5, 16)
        self.normalize = nn.LayerNorm(16)

        encoder_layer = nn.TransformerEncoderLayer(d_model=16, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.head = nn.Linear(16, 1)


    def forward(self, x):

        batch_size, _ = x.shape

        # returns 6, 2 tensor
        e_ids = self.exercise_embedding(self.exercise_ids)

        # returns 6, 2 tensor
        s_ids = self.seq_embedding(self.seq_ids)

        emb = torch.cat([e_ids, s_ids], dim=1)
        emb = emb.unsqueeze(0)
        emb = emb.expand(batch_size, -1, -1)
        reps = x.unsqueeze(-1)

        x = torch.cat([emb, reps], dim=2)
        x = self.input_projection(x)
        x = self.normalize(x)
        x = self.transformer(x)
        logits = self.head(x).squeeze(-1)

        return logits
