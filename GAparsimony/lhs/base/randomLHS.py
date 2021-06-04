# -*- coding: utf-8 -*-

from GAparsimony.lhs.util import findorder, order
import numpy as np

   
def randomLHS(n, k, bPreserveDraw=False, seed=None):
    
    if n < 1 or k < 1:
        raise Exception("nsamples are less than 1 (n) or nparameters less than 1 (k)")

    if seed:
        np.random.seed(seed)

    result = np.zeros(n*k).reshape((n, k))
        
    randomunif1 = np.empty(n).astype(np.double)

    if bPreserveDraw:
        
        randomunif2 = np.empty(n).astype(np.double)
        for jcol in range(k):
            
            for irow in range(n):
                randomunif1[irow] = np.random.uniform(low=0, high=1)
            for irow in range(n):
                randomunif2[irow] = np.random.uniform(low=0, high=1)


            orderVector = order(randomunif1)
            for irow in range(n):
                result[irow,jcol] = orderVector[irow] + randomunif2[irow]
                result[irow,jcol] =  result[irow,jcol] / np.double(n)

    else:
        randomunif2 = np.empty(n*k).astype(np.double)
        for jcol in range(k):
            for irow in range(n):
                randomunif1[irow] = np.random.uniform(low=0, high=1)

            orderVector = order(randomunif1)
            for irow in range(n):
                result[irow,jcol] = orderVector[irow]

        for i in range(n*k):
            randomunif2[i] = np.random.uniform(low=0, high=1)

        randomunif2 = randomunif2.reshape((n, k))
        for jcol in range(k):

            for irow in range(n):
                result[irow,jcol] = result[irow,jcol] + randomunif2[irow, jcol]
                result[irow,jcol] = result[irow,jcol] / np.double(n)

    return result

def randomLHS_int(n, k, seed=None):

    if seed:
        np.random.seed(seed)

    result = np.empty((n, k)).astype(np.int32)
    randomunif1 = np.empty(n).astype(np.double)
    for jcol in range(k):
    
        for irow in range(n):
            randomunif1[irow] = np.random.uniform(low=0, high=1)
        
        orderVector = findorder(randomunif1)
        for irow in range(n):
            result[irow,jcol] = orderVector[irow]

    return result
