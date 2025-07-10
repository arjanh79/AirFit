from v1.AI.biLSTM import AirFitBiLSTM
from v1.AI.data_preprocessing import WorkoutPreprocessor
from v1.AI.dataset import WorkoutDataset
from v1.AI.train import ModelTraining

pre = WorkoutPreprocessor()

dataset = WorkoutDataset(pre.embeddings_x, pre.data_x, pre.data_y, pre.weighted_loss)

mt = ModelTraining(AirFitBiLSTM(), dataset)

mt.train_model()
mt.eval_model()

mt.make_pred()