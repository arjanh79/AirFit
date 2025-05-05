
import numpy as np

wl = np.array([[0.0364, 0.0729, 0.1566, 0.2530, 0.3638, 0.4912, 0.6377, 0.8062, 1.0000]])
y_hat = np.array([[3.2342, 3.4660, 2.9589, 4.9412, 4.3478, 3.9519, 3.5844, 3.0091, 3.1857]])
y_true = np.array([[4., 3., 2., 5., 4., 4., 3., 3., 3.]])

# MSE

loss = (y_hat - y_true) ** 2
mean = np.mean(loss * wl)

print(mean)