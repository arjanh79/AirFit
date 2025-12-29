from torch.utils.data import DataLoader

from v2.Data.factories import RepositoryFactory
from v2.Data.intensity_dataset import IntensityDataset
from v2.Domain.intensity_combinator import IntensityCombinator


def main() -> None:

    repo = RepositoryFactory.get_repository('sqlite')
    combinator = IntensityCombinator()
    workouts = combinator.get_data()
    ds = IntensityDataset(workouts)
    dl = DataLoader(ds, batch_size=32, shuffle=True)

    embedding_dims = {
        'exercise_id': len(repo.get_exercise_ids()[0]),
        'weight_id': len(repo.get_weight_ids()[0]),
        'equipment_id': len(repo.get_equipment_ids()[0]),
        'core': 2,
        'exercise_sequence': 6,
        'metric_type': 2
    }



if __name__ == "__main__":
    main()