from v2.Trainer.trainer import WorkoutTrainer
from v2.config import MODEL_PATH
from v2.Domain.workout_combinator import WorkoutCombinator
from v2.Data.workout_dataset import WorkoutDataset

import json


def main() -> None:

    combinator = WorkoutCombinator()
    workouts = combinator.create_workouts()
    ds = WorkoutDataset(workouts)

    save_token_mappings('ex_to_id.json', ds.ex_to_id)
    save_token_mappings('id_to_ex.json', ds.id_to_ex)

    trainer = WorkoutTrainer(combinator, ds)
    trainer.fit(epochs=50)
    trainer.save_model()


def save_token_mappings(filename: str, data: dict) -> None:
    with open(MODEL_PATH / filename, 'w') as f:
        json.dump(data, f)



if __name__ == "__main__":
    main()