
import torch

wl = torch.tensor([0.0364, 0.0729, 0.1566, 0.2530, 0.3638, 0.4912, 0.6377, 0.8062, 1.0000])
y_hat = torch.tensor([3.0504, 3.4092, 3.0151, 4.2135, 3.7287, 3.5623, 3.1515, 2.8245, 3.1567])
y_true = torch.tensor([4., 3., 2., 5., 4., 4., 3., 3., 3.])


loss_fn = torch.nn.MSELoss(reduction='none')
loss = loss_fn(y_hat, y_true) * wl

loss = torch.mean(loss)

print(loss.item())