import torch
import torch.nn as nn

class IntensityTransformer(nn.Module):
    def __init__(self, embeddings_dim: dict[str, int], col_names: list[str]):
        super().__init__()

        # {'exercise_id': 25, 'weight_id': 16, 'equipment_id': 6, 'core': 2, 'exercise_sequence': 6, 'metric_type': 2}
        self.embeddings_dim = embeddings_dim
        self.col_names = col_names
        print(self.col_names)
        self.d_model = 8

        self.feature_cfg = {
            'exercise_id': (self.d_model, 0),
            'exercise_sequence': (2, 0),
            'weight_id': (4, 0),
            'core': (2, None),
            'metric_type': (2, None),
            'equipment_id': (4, 0),
        }
        self.emb = nn.ModuleDict({
            name: nn.Embedding(
                num_embeddings=self.embeddings_dim[name],
                embedding_dim=emb_dim,
                padding_idx=pad_idx
            )
            for name, (emb_dim, pad_idx) in self.feature_cfg.items()
        })
        self.proj = nn.ModuleDict({
            name: nn.Identity() if emb_dim == self.d_model else nn.Linear(emb_dim, self.d_model)
            for name, (emb_dim, _) in self.feature_cfg.items()
        })


    def forward(self, x):
        # self.feature_cols = ['exercise_id', 'exercise_sequence', 'weight_id', 'reps', 'core', 'metric_type', 'equipment_id']
        pass


