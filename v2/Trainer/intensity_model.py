import torch
import torch.nn as nn

class IntensityTransformer:
    def __init__(self, d_model: int = 8, n_head: int = 4, num_layers: int = 2):

        self.d_model = d_model  # dimension of the embedding
        self.n_head = n_head    # number of attention heads
        self.num_layers = num_layers # Number of sub-encoder layers

        enc_layer = nn.TransformerEncoderLayer(d_model, nhead=self.n_head, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.lm_head = nn.Linear(d_model, 1, bias=True) # Final output of the model


    def forward(self):
        pass


