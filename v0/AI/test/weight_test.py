import numpy as np
import torch


weights = torch.tensor(np.cumprod([1.0075] * 10), dtype=torch.float32)
print(weights)

