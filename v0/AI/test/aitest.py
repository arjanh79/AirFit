from dev.v0.AI.data_preprocessing import WorkoutPreprocessor
from dev.v0.AI.dataset import WorkoutDataset
from dev.v0.AI.modelDNN import AirFitDNN
from dev.v0.AI.train import ModelTraining

pre = WorkoutPreprocessor()
dataset = WorkoutDataset(pre.embeddings_x, pre.data_x, pre.data_y, pre.weighted_loss)


mt = ModelTraining(AirFitDNN(), dataset)

mt.train_model()