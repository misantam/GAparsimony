from lhs.base.randomLHS import randomLHS, randomLHS_int
from lhs.base.maximinLHS import maximinLHS
from lhs.base.improvedLHS import improvedLHS
from lhs.base.optimumLHS import optimumLHS
from lhs.base.geneticLHS import geneticLHS
from lhs.util.utilityLHS import isValidLHS, isValidLHS_int

import numpy as np


aux = randomLHS_int(2,2)
print(aux)
print(isValidLHS_int(aux))


# m = randomLHS(4,5)

# m = np.array([[0.04655382, 0.4243821], [0.92951608, 0.9661060]])

# randomLHS
# m = randomLHS(3,6)

# m[0, 0] = 0.6

# print(m)

# maximinLHS

m = np.array([[0.6062350, 0.4130588], [0.3730882, 0.7029388]])

m = maximinLHS(2, 2) ## Esta dando valores enteros revisar, la de arriba es correcta

# m[0, 0] = 0.6

print(m)

print(isValidLHS(m))

# print(np.finfo(np.double).tiny) # 2.22507e-308

# p = np.random.rand(2,2)
# d = squareform(pdist(p, 'euclidean'))

# print(d)

# print(np.tril(d).flatten())
# aux = np.tril(d).flatten()
# aux = aux[aux!=0]
# print(aux)
# print(np.min(aux))

m = improvedLHS(2, 2) ## Esta dando valores enteros revisar, la de arriba es correcta

# m[0, 0] = 0.6

print(m)

print(isValidLHS(m))


m = optimumLHS(2, 2) ## Da errores con dimensión superior a 2x2

# m[0, 0] = 0.6

print(m)

print(isValidLHS(m))

# m = geneticLHS(2, 2) ## Esta dando valores enteros revisar, la de arriba es correcta
m = geneticLHS(2, 2) ## Da errores con dimensión superior a 2x2

# m[0, 0] = 0.6

print(m)

print(isValidLHS(m))