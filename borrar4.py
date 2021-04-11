import numpy as np

a = np.empty(5, dtype=np.object)

print(a)

a[0] = np.array([1,2,3])

print(a)

a[2] = np.array([1,2])

print(a)


min_param = np.array([00.0001, 0.00001])
max_param = np.array([99.9999, 0.99999])

print(np.stack([min_param, max_param], axis=0))