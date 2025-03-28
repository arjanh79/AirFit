import torch
import torch.nn as nn

class AirFitDNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(13, 3)
        self.seq_features = nn.Linear(60, 15)

        self.fc1 = nn.Linear(75, 25)
        self.relu1 = nn.LeakyReLU()
        self.fc2 = nn.Linear(25, 10)
        self.relu2 = nn.LeakyReLU()
        self.output = nn.Linear(10, 1)

        with torch.no_grad():
            self.embedding.weight[0] = torch.zeros(3)

    def forward(self, e, f):
        e = self.embedding(e)
        e = e.flatten(start_dim=1)
        f = self.seq_features(f)
        x = torch.cat((e, f), dim=1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.output(x)

        return x
