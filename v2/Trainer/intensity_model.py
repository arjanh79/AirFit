import torch
import torch.nn as nn

class IntensityTransformer(nn.Module):
    def __init__(self, embeddings_dim: dict[str, int]):
        super().__init__()

        # {'exercise_id': 25, 'weight_id': 16, 'equipment_id': 6, 'core': 2, 'exercise_sequence': 6, 'metric_type': 2}
        self.embeddings_dim = embeddings_dim
        self.d_model = 8

        # overrides of self.d_model
        self.d_weight = 4
        self.d_seq = 2
        self.d_equipment = 4
        self.d_core = 2
        self.d_metric = 2

        self.exercise_emb = nn.Embedding(self.embeddings_dim['exercise_id'], self.d_model, padding_idx=0)
        self.weight_emb = nn.Embedding(self.embeddings_dim['weight_id'], self.d_weight, padding_idx=0)
        self.seq_emb = nn.Embedding(self.embeddings_dim['exercise_sequence'], self.d_seq, padding_idx=0)
        self.equipment_emb = nn.Embedding(self.embeddings_dim['equipment_id'], self.d_seq, padding_idx=0)
        self.core_emb = nn.Embedding(self.embeddings_dim['core'], self.d_core, padding_idx=None)
        self.metric_emb = nn.Embedding(self.embeddings_dim['metric_type'], self.d_metric, padding_idx=None)

        self.weight_fc = nn.Linear(self.d_weight, self.d_model)
        self.seq_fc = nn.Linear(self.d_seq, self.d_model)
        self.equipment_fc = nn.Linear(self.d_equipment, self.d_model)
        self.core_fc = nn.Linear(self.d_core, self.d_model)
        self.metric_fc = nn.Linear(self.d_metric, self.d_model)

    def forward(self):
        pass


