from v2.Data.factories import RepositoryFactory
from v2.Trainer.trainer import WorkoutTrainer
from v2.config import MODEL_PATH
from v2.Domain.workout_combinator import WorkoutCombinator
from v2.Data.workout_dataset import WorkoutDataset

import json


def main() -> None:

    repo = RepositoryFactory.get_repository('sqlite')

    combinator = WorkoutCombinator()
    workouts = combinator.create_workouts()
    ds = WorkoutDataset(workouts)

    exercise_mappings, _ = repo.get_exercise_ids()
    exercise_mappings = {k: v for k, v in exercise_mappings}

    save_data('ex_to_id.json', ds.ex_to_id)
    save_data('id_to_ex.json', ds.id_to_ex)
    save_data('exercise_ids.json', exercise_mappings)

    trainer = WorkoutTrainer(combinator, ds)
    trainer.fit(epochs=5)
    trainer.save_model()


def save_data(filename: str, data: dict) -> None:
    with open(MODEL_PATH / filename, 'w') as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    main()