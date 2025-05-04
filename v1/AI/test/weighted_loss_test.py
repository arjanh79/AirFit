
import numpy as np
import torch

sample_length = 10 - 1

weighted_loss = torch.tensor(np.cumprod([1.15] * sample_length), dtype=torch.float32)
weighted_loss = torch.cat((torch.ones(1), weighted_loss))
weighted_loss = (weighted_loss - weighted_loss.min()) / (weighted_loss.max() - weighted_loss.min())
weighted_loss = weighted_loss.reshape((-1, 1))
weighted_loss[0] = weighted_loss[1] / 2

print(weighted_loss)