
import torch
import torch.nn as nn

class WorkoutTransformer(nn.Module):

    def __init__(self, vocab_size: int, d_model: int = 8, n_head: int = 4, num_layers: int = 2, max_len: int = 13):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_head = n_head

        self.num_layers = num_layers
        self.max_len = max_len
        self.pad_idx = 0

        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=self.pad_idx)
        self.pos_emb = nn.Embedding(max_len, d_model)

        enc_layer = nn.TransformerEncoderLayer(d_model, nhead=self.n_head, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t = x.shape   # batch_size, number of tokens in sequence
        pos = torch.arange(t).unsqueeze(0).expand(b, t)

        h = self.tok_emb(x) + self.pos_emb(pos)

        causal_mask = torch.triu(torch.ones(t, t, dtype=torch.bool), diagonal=1) # Disallow attending to future tokens
        pad_mask = (x == self.pad_idx) # optional padding mask (True = pad positions to ignore)

        h = self.encoder(h, mask=causal_mask, src_key_padding_mask=pad_mask)
        logits = self.lm_head(h)

        return logits
