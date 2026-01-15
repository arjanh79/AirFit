
import torch

class IntensityRounder:

    def __init__(self, ig):

        self.repo = ig.repo
        self.model = ig.model

        self.x = ig.ds[0][0].unsqueeze(0).clone()

        self.reps = ig.reps.clone()
        self.reps_col = ig.feature_cols.index('reps')

    def apply(self):

        exercise_ids = self.exercise_ids()
        default_steps = self.default_steps(exercise_ids)
        return self.round_reps(default_steps)

    def exercise_ids(self):
        return self.x[:, :, 0].to(torch.int64).tolist()[0]

    def default_steps(self, exercise_ids):
        data, _ = self.repo.get_exercise_steps()
        step_map = dict(data)
        steps = [step_map[eid] for eid in exercise_ids]
        return torch.tensor(steps, dtype=torch.float32)

    def round_reps(self, default_steps):
        reps = (self.reps // default_steps) * default_steps

        x_reps = self.x.clone()
        x_reps[:, :, self.reps_col] = reps

        intensity = self.model(x_reps).item()

        if intensity < 4.5:
            reps += default_steps
            x_reps[:, :, self.reps_col] = reps
            intensity = self.model(x_reps).item()

        return reps.int(), intensity




