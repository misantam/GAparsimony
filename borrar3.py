import numpy as np

m1 = np.array([0, 1, 2, 3, 4])
m2 = np.array([5, 4, 2, 3, 1])

init = 0
op1 = lambda a, b: a + b
op2 = lambda a, b: a==b

for x, y in zip(m1, m2):
    init = op1(init, op2(x, y))

print(init) #2

aux = np.array([72, 69, 76, 76, 79])

# def transform(a, b= None, dev = 0, op=lambda x: x):
#     if not type(b) is type(None):
#         for x, y in zip(a, b):
#             dev = dev + 1
#             dev = op(x, y)
#     else:
#         for x in a:
#             dev = dev + 1
#             dev = op(x)
#     return dev

# print(transform(aux, aux, lambda a, b: a+b))

print(list(map(lambda a, b: a+b, aux, aux)))

# Transform es map


aux = np.array([[2,1],[2,1]])

print(aux[0, 0])

