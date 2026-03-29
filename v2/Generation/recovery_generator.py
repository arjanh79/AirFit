import torch

from v2.Data.factories import RepositoryFactory
from v2.Data.recovery_dataset import RecoveryDataset
from v2.Domain.recovery_combinator import RecoveryCombinator
from v2.Trainer.Recovery.recovery_trainer import RecoveryTransformer
from v2.config import MODEL_PATH


class RecoveryGenerator:
    def __init__(self):
        self.feature_cols = [
                    'exercise_id', 'exercise_sequence', 'weight_id', 'reps',
                    'core', 'metric_type', 'equipment_id'
                    ]

        self.repo = RepositoryFactory.get_repository('sqlite')
        self.combinator = RecoveryCombinator()
        self.workout_id = self.get_workout_id()

        self.ds = RecoveryDataset(self.combinator.get_data(completed = False))

        self.model = self.rebuild_model()


    def eval(self):
        with torch.no_grad():
            x, y, l = self.ds[0]
            x = x.unsqueeze(0).to(torch.float32)
            probs = self.model(x)
            return torch.nn.Sigmoid()(probs).item()

    def get_workout_id(self):
        data, cols = self.repo.check_available_workout()
        return data[0][0]


    def rebuild_model(self):
        num_embedding = {
            'exercise_id': len(self.repo.get_exercise_ids()[0]) + 2,
            'weight_id': len(self.repo.get_weight_ids()[0]) + 1,
            'equipment_id': len(self.repo.get_equipment_ids()[0]) + 1,
            'core': 2,
            'exercise_sequence': 6 + 1,
            'metric_type': 2
        }

        model = RecoveryTransformer(num_embeddings = num_embedding, col_names=self.feature_cols)

        best_model = torch.load(MODEL_PATH / 'recovery_model_best.pth')
        model.load_state_dict(best_model['model_state'])
        model.eval()
        return model
