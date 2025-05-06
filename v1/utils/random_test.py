import numpy as np

rng = np.random.default_rng()

i = rng.normal(3, 0.25, 100)

print(np.sort(i))