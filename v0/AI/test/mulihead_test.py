import torch
import torch.nn as nn

class AirFitMultiHeadDNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.heads = 20 # Each exercise in the workout gets a head
        self.num_exercises = 13 # 13 difference exercises
        self.embeddings_dim = 3 # Use a 3D representation per exercise

        self.embedding = nn.Embedding(self.num_exercises, self.embeddings_dim)
        self.features = nn.Linear(3, 5)  # Increase number of features, interactions

        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(8, 10),
                nn.LeakyReLU(),
                nn.Linear(10, 1),
                nn.Softplus()
            ) for _ in range(self.heads)
        ])
        self.output_layer = nn.Linear(20, 1)
        self.final_relu = nn.ReLU()

        with torch.no_grad():
            self.embedding.weight[0] = torch.zeros(self.embeddings_dim)

    def forward(self, e, f):
        print(f)
        e = self.embedding(e)
        f = f.reshape((-1, 20, 3)) # Reshape f, create a 3 vector per exercise
        print(f)
        f = self.features(f)
        x = torch.cat((e, f), dim=2) # Output: torch.Size([2, 20, 8])
        h = [head(x[:, i, :]) for i, head in enumerate(self.heads)]
        h = torch.cat(h, dim=1)
        print(h)
        output = self.output_layer(h)
        # print(output)
        return output

model = AirFitMultiHeadDNN()
print(model)
