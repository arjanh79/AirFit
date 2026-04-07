from dataclasses import dataclass

@dataclass(frozen=True)
class WorkoutBody:
    exercise_id: int
    exercise_sequence: int
    core: int
    reps: int
    weight_id: int

@dataclass(frozen=True)
class WorkoutHeader:
    workout_id: str
    workout_intensity: float
    train_tomorrow: int
    active_kcal: int
    loss: float

@dataclass
class Workout():
    header: WorkoutHeader
    body: list[WorkoutBody]