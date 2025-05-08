import time

import numpy as np

arr = np.random.rand(1920, 1080, 3, 250)
t = time.time()
for i in range(100000):
    it = arr[i % len(arr)]
print(time.time() - t)
