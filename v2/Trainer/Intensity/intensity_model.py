import torch
import torch.nn as nn

class IntensityTransformer(nn.Module):
    def __init__(self, num_embeddings: dict[str, int], col_names: list[str]):
        super().__init__()

        self.num_embeddings = num_embeddings  # vocab_size
        self.col_names = col_names
        self.col_mappings = {v: c for c, v in enumerate(self.col_names)}

        self.d_model = 8
        self.n_head = 4
        self.num_layers = 2


        self.feature_cfg = {
            'exercise_id':       {'type': 'embed', 'emb_dim': self.d_model, 'pad': 0},
            'weight_id':         {'type': 'embed', 'emb_dim': 4,            'pad': 0},
            'exercise_sequence': {'type': 'embed', 'emb_dim': 2,            'pad': 0},
            'equipment_id':      {'type': 'embed', 'emb_dim': 4,            'pad': 0},
            'core':              {'type': 'embed', 'emb_dim': 2,            'pad': None},
            'metric_type':       {'type': 'embed', 'emb_dim': 2,            'pad': None},

            'reps': {'type': 'numeric'}
        }

        self.emb = nn.ModuleDict({
            name: nn.Embedding(
                num_embeddings=self.num_embeddings[name],
                embedding_dim=cfg['emb_dim'],
                padding_idx=cfg.get('pad', None)
            )
            for name, cfg in self.feature_cfg.items()
            if cfg['type'] == 'embed'
        })
        self.proj = nn.ModuleDict()
        for name, cfg in self.feature_cfg.items():
            if cfg['type'] == 'numeric':
                self.proj[name] = nn.Linear(1, self.d_model, bias=False)
            else:
                emb_dim = cfg['emb_dim']
                self.proj[name] = nn.Identity() if emb_dim == self.d_model else nn.Linear(emb_dim, self.d_model)

        enc_layer = nn.TransformerEncoderLayer(self.d_model, nhead=self.n_head, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=self.num_layers)
        self.lm_head = nn.Linear(self.d_model, 1, bias=True)


    def forward(self, x):

        b, t, f = x.shape
        embeddings_all = 0

        for name in self.col_names:
            values = x[:, :, self.col_mappings[name]]
            if self.feature_cfg[name]['type'] == 'numeric':
                v = torch.log1p(values.float()).unsqueeze(-1)
                embeddings_all += self.proj[name](v)
            else:
                v = self.emb[name](values.long())
                embeddings_all += self.proj[name](v)

        causal_mask = torch.triu(torch.ones(t, t, dtype=torch.bool), diagonal=1)
        h = self.encoder(embeddings_all, mask=causal_mask)
        logits = self.lm_head(h) # (B, T, 1) intensity per exercise
        output = torch.sum(logits, dim=1)  # (B, 1) workout intensity from exercises
        return output

