
import numpy as np
import torch

import matplotlib.pyplot as plt

sample_length = 50 - 1

weighted_loss = torch.tensor(np.cumprod([1.15] * sample_length), dtype=torch.float32)
weighted_loss = torch.cat((torch.ones(1), weighted_loss))
weighted_loss = (weighted_loss - weighted_loss.min()) / (weighted_loss.max() - weighted_loss.min())
# weighted_loss = weighted_loss

print(weighted_loss)

plt.plot(weighted_loss)
plt.show()