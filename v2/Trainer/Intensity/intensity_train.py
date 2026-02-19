
from v2.Data.factories import RepositoryFactory
from v2.Data.intensity_dataset import IntensityDataset
from v2.Domain.intensity_combinator import IntensityCombinator
from v2.Trainer.Intensity.intensity_trainer import IntensityTrainer


def main() -> None:

    repo = RepositoryFactory.get_repository('sqlite')
    combinator = IntensityCombinator()
    workouts = combinator.get_data(completed = True)
    ds = IntensityDataset(workouts)

    num_embedding = {
        'exercise_id': 26, #len(repo.get_exercise_ids()[0]) + 1,
        'weight_id': len(repo.get_weight_ids()[0]) + 1,
        'equipment_id': len(repo.get_equipment_ids()[0]) + 1,
        'core': 2,
        'exercise_sequence': 6 + 1,
        'metric_type': 2
    }

    trainer = IntensityTrainer(combinator=combinator, dataset=ds, num_embeddings=num_embedding, col_names=ds.feature_cols)
    trainer.fit(epochs=500)
    trainer.save_model('last')


if __name__ == "__main__":
    main()