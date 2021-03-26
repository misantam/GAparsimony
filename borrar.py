# from concurrent.futures import ThreadPoolExecutor, as_completed


# def func(a,b):
#     return (a+1, b+2)

# def paralel_func(alist, blist):
#     processes = []
#     with ThreadPoolExecutor(max_workers=None) as executor:
#         processes.append(executor.map(func, alist, blist))

#         for _ in as_completed(processes):
#             print('Result: ', _.result())


# paralel_func(list(range(5)), list(range(5,10)))

import numpy as np

# aux = np.array(range(100))
# aux = aux.astype(float)

# aux[10] = np.nan
# aux[20] = np.nan
# aux[30] = np.nan
# aux[40] = np.nan
# aux[50] = np.nan
# aux[60] = np.nan
# aux[70] = np.nan
# aux[80] = np.nan
# aux[90] = np.nan


# print(aux)
# aux[np.isnan(aux)]=- np.float32("inf")
# print(aux)

aux = np.array([5,8,655,4,52,5,4,np.nan, 4,566,5,4,222,11,2])

# print(aux[-2:])

# print(np.count_nonzero(np.isnan(aux)))

# print(np.argsort(aux, kind='heapsort')[::-1])



# print(aux[np.argsort(aux)[::-1]])


# print(aux[aux.argsort(kind='heapsort')])

# def order(obj, kind='heapsort', decreasing = False, na_last = True):
#     if not decreasing:
#         if na_last:
#             return obj.argsort(kind=kind)
#         else:
#             na = np.count_nonzero(np.isnan(obj))
#             aux = obj.argsort(kind=kind)
#             return np.concatenate([aux[-na:], aux[:-na]])
#     else:
#         if not na_last:
#             return obj.argsort(kind=kind)[::-1]
#         else:
#             na = np.count_nonzero(np.isnan(obj))
#             aux = obj.argsort(kind=kind)[::-1]
#             return np.concatenate([aux[na:], aux[:na]])



# print(aux[order(aux, kind='heapsort', decreasing = False, na_last = True)])
# print(len(aux))



m = np.random.normal(0,1,30)
print(m)

sampl = np.random.uniform(low=0, high=1)
print(sampl)

# print((np.random.rand((5*10,1)) * (10 - 5) + 5).shape)

print(np.random.uniform(low=0, high=1))

a=np.array([[1,2,3],[1,2,3],[1,2,3]])

print(np.empty(a.shape, dtype=object).dtype)


aux = np.array([(3,5), (1,1), (2,4)])

# print(list(aux))
# l = list(aux)
# sorted(list(aux), key=lambda a: a[0])

print(sorted(aux, key=lambda a: a[0]))

print(np.empty(5))

print(np.floor(np.double(2.3)))