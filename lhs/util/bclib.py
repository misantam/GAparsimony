import numpy as np

# Ya estaría implementadas

def findorder_zero(v):
    
    # create a vector of pairs to hold the value and the integer rank
    p = np.empty(v.shape, dtype=object)
    for i, vi in enumerate(v):
        p[i] = (vi, i)
    p = np.array(list(sorted(p, key=lambda a: a[0])))
    
    return np.array(list(map(lambda x: x[1], p)))

def order(obj, kind='heapsort', decreasing = False, na_last = True):
    if not decreasing:
        if na_last:
            return obj.argsort(kind=kind)
        else:
            na = np.count_nonzero(np.isnan(obj))
            aux = obj.argsort(kind=kind)
            return np.concatenate([aux[-na:], aux[:-na]])
    else:
        if not na_last:
            return obj.argsort(kind=kind)[::-1]
        else:
            na = np.count_nonzero(np.isnan(obj))
            aux = obj.argsort(kind=kind)[::-1]
            return np.concatenate([aux[na:], aux[:na]])



def findorder(v):
    order = findorder_zero(v)
    for i in range(len(order)):
        order[i] += 1
    return order
 

def inner_product(a, b, init=0, op1=lambda a, b: a + b, op2=lambda a, b: a * b):
    for x, y in zip(a, b):
        init = op1(init, op2(x, y))
    return init
