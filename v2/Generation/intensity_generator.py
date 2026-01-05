import numpy as np
import torch

from v2.Data.factories import RepositoryFactory
from v2.Data.intensity_dataset import IntensityDataset
from v2.Domain.intensity_combinator import IntensityCombinator
from v2.Generation.intensity_round import IntensityRounder
from v2.Generation.workout_generator import WorkoutGenerator
from v2.Trainer.Intensity.intensity_model import IntensityTransformer
from v2.config import MODEL_PATH


class IntensityGenerator:
    def __init__(self):
        self.feature_cols = [
                    'exercise_id', 'exercise_sequence', 'weight_id', 'reps',
                    'core', 'metric_type', 'equipment_id'
                    ]

        self.repo = RepositoryFactory.get_repository('sqlite')
        self.combinator = IntensityCombinator()
        self.workout_generator = WorkoutGenerator()
        self.workout_id = self.get_workout_id()

        self.ds = IntensityDataset(self.combinator.get_data(completed = False))

        self.model = self.rebuild_model()
        self.reps = self.reps_gradient()

        self.new_reps, self.intensity = IntensityRounder(self).apply()
        print('\nNEW REPS:', self.new_reps.tolist())
        self.update_reps()
        self.update_intensity()


    def get_workout_id(self):
        data, cols = self.repo.check_available_workout()
        return data[0][0]

    def update_reps(self):
        x = self.ds[0][0].clone()

        reps_col = self.feature_cols.index('reps')
        core_col = self.feature_cols.index('core')
        seq_col = self.feature_cols.index('exercise_sequence')

        x = x[:, [core_col, seq_col, reps_col]]
        x[:, 2] = self.new_reps

        self.repo.update_reps(self.workout_id, x)

    def update_intensity(self):
        self.repo.update_intensity(self.workout_id, np.round(self.intensity, decimals=2))



    def refresh_data(self) -> None:
        data, _ = self.repo.check_available_workout()
        if len(data) == 0:
            self.workout_generator.generate()


    def rebuild_model(self):
        num_embedding = {
            'exercise_id': len(self.repo.get_exercise_ids()[0]) + 1,
            'weight_id': len(self.repo.get_weight_ids()[0]) + 1,
            'equipment_id': len(self.repo.get_equipment_ids()[0]) + 1,
            'core': 2,
            'exercise_sequence': 6 + 1,
            'metric_type': 2
        }

        model = IntensityTransformer(num_embeddings = num_embedding, col_names=self.feature_cols)
        model.load_state_dict(torch.load(MODEL_PATH / f'intensity_model_best.pth', weights_only=True))
        model.eval()
        return model


    def reps_gradient(self):
        x, y, l = self.ds[0]
        x = x.unsqueeze(0).to(torch.float32)

        reps_col = self.feature_cols.index('reps')
        reps = torch.nn.Parameter(x[0, :, reps_col].clone())
        optimizer = torch.optim.NAdam([reps], lr=0.03)
        target = torch.tensor(4.5)

        loss_fn = torch.nn.MSELoss()

        for step in range(100 + 1):
            optimizer.zero_grad()

            x_work = x.clone()
            x_work[0, :, reps_col] = reps

            out = self.model(x_work).squeeze()

            if 4.25 <= out.item() <= 4.75:  # Good enough
                break

            loss = loss_fn(out, target)
            loss.backward()

            optimizer.step()

            with torch.no_grad():
                reps.clamp_(10, 120)

            if step % 50 == 0:
                print(f'step {step:04d}: intensity={out.item():.5f} reps={reps.round().int().tolist()}')
        return reps




ig = IntensityGenerator()
ig.refresh_data()
