import torch
import torch.nn as nn

class AirFitDNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(13, 3)
        self.seq_features = nn.Linear(79, 5)
        self.fc1 = nn.Linear(5 + 3, 10)
        self.relu = nn.LeakyReLU()
        self.output = nn.Linear(10, 1)


    def forward(self, e, f):
        e = self.embedding(e)
        e = e.flatten(start_dim=1)

        f = self.seq_features(f)

        x = torch.cat((e, f), dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.output(x)

        return x