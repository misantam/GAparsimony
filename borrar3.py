import numpy as np

population = np.array(list(range(50,100)))

popSize = 50
r = 2/(popSize*(popSize-1))
q = 2/popSize
rank = list(range(popSize))
prob = list(map(lambda x: q - (x)*r, rank))

sel = np.random.choice(list(rank), size=popSize, replace=True, p=list(map(lambda x: np.min(np.ma.masked_array(np.array([max(0, x), 1]), np.isnan(np.array([max(0, x), 1])))), prob)))

print(sel)

print(population[sel])

print(population[:5])