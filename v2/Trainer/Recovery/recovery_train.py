

from v2.Data.factories import RepositoryFactory
from v2.Data.recovery_dataset import RecoveryDataset
from v2.Domain.recovery_combinator import RecoveryCombinator
from v2.Trainer.Recovery.recovery_trainer import RecoveryTrainer

def main() -> None:

    repo = RepositoryFactory.get_repository('sqlite')
    combinator = RecoveryCombinator()
    workouts = combinator.get_data()
    ds = RecoveryDataset(workouts)

    num_embedding = {
        'exercise_id': len(repo.get_exercise_ids()[0]) + 2,
        'weight_id': len(repo.get_weight_ids()[0]) + 1,
        'equipment_id': len(repo.get_equipment_ids()[0]) + 1,
        'core': 2,
        'exercise_sequence': 6 + 1,
        'metric_type': 2
    }

    trainer = RecoveryTrainer(combinator=combinator, dataset=ds, num_embeddings=num_embedding, col_names=ds.feature_cols)
    trainer.fit(epochs=5000)
    trainer.save_model('last')


if __name__ == "__main__":
    main()