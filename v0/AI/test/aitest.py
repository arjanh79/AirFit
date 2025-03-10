from dev.v0.AI.data_preprocessing import WorkoutPreprocessor
from dev.v0.AI.dataset import WorkoutDataset
from dev.v0.AI.modelDNN import AirFitDNN
from dev.v0.AI.train import train_model, eval_model

pre = WorkoutPreprocessor()
dataset = WorkoutDataset(pre.embeddings_x, pre.data_x, pre.data_y, pre.weighted_loss)

model = AirFitDNN()

# train_model(model, dataset)
eval_model(model, dataset)