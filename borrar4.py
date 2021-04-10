import numpy as np

a = np.empty(5, dtype=np.object)

print(a)

a[0] = np.array([1,2,3])

print(a)

a[2] = np.array([1,2])

print(a)