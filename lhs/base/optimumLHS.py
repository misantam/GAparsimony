# -*- coding: utf-8 -*-

import numpy as np
from lhs.util.bclib import findorder
from lhs.util.utilityLHS import sumInvDistance, convertIntegerToNumericLhs


def _optimumLHS(n, k, optimalityRecordLength, maxSweeps=2, eps=0.1):
    nOptimalityRecordLength = optimalityRecordLength
    nsamples = n
    nparameters = k
    nMaxSweeps = maxSweeps
    eps_change = eps
    extraColumns = 0
    optimalityChangeOld = 0.0
    outlhs = np.empty((nsamples, nparameters))
    newHypercube = np.empty((nsamples, nparameters))
    optimalityRecord = np.empty(nOptimalityRecordLength).astype(np.double)
    interchangeRow1 = np.zeros(nOptimalityRecordLength)
    interchangeRow2 = np.zeros(nOptimalityRecordLength)

    #  fill the oldHypercube with a random lhs sample
    randomUnif = np.empty(nsamples).astype(np.double)
    for jcol in range(nparameters):
        #  fill a vector with a random sample to order
        for irow in range(nsamples):
            randomUnif[irow] = np.random.uniform(low=0, high=1)
        
        orderedUnif = findorder(randomUnif)
        for irow in range(nsamples):
            outlhs[irow,jcol] = orderedUnif[irow]
    
    # find the initial optimality measure 
    gOptimalityOld = sumInvDistance(outlhs)
    test = 0
    iter = 0
    while test == 0:
    
        if iter == nMaxSweeps:
            break
        iter += 1
        # iterate over the columns 
        for j in range(nparameters):
        
            optimalityRecordIndex = 0
            # iterate over the rows for the first point from 0 to N-2 
            for i in range(nsamples - 1):
            
                # iterate over the rows for the second point from i+1 to N-1 
                for kindex in range((i + 1), nsamples):
                
                    # put the values from oldHypercube into newHypercube 
                    newHypercube = outlhs.copy()
                    # exchange two values (from the ith and kth rows) in the jth column
                    # * and place them in the new matrix 
                    newHypercube[i, j] = outlhs[kindex, j]
                    newHypercube[kindex, j] = outlhs[i, j]
                    # store the optimality of the newly created matrix and the rows that
                    # * were interchanged 
                    optimalityRecord[optimalityRecordIndex] = sumInvDistance(newHypercube)
                    interchangeRow1[optimalityRecordIndex] = i
                    interchangeRow2[optimalityRecordIndex] = kindex
                    optimalityRecordIndex += 1

            # once all combinations of the row interchanges have been completed for
            # * the current column j, store the old optimality measure (the one we are
            # * trying to beat) 
            optimalityRecord[optimalityRecordIndex] = gOptimalityOld
            interchangeRow1[optimalityRecordIndex] = 0
            interchangeRow2[optimalityRecordIndex] = 0
            # Find which optimality measure is the lowest for the current column.
            # * In other words, which two row interchanges made the hypercube better in
            # * this column 
            posit = 0
            for kindex in range(nOptimalityRecordLength):
            
                if optimalityRecord[kindex] < optimalityRecord[posit]:
                    posit = kindex
            # If the new minimum optimality measure is better than the old measure 
            if optimalityRecord[posit] < gOptimalityOld:
            
                # put oldHypercube in newHypercube 
                newHypercube = outlhs.copy()
                # Interchange the rows that were the best for this column 
                newHypercube[int(interchangeRow1[posit]), j] = outlhs[int(interchangeRow2[posit]), j]
                newHypercube[int(interchangeRow2[posit]), j] = outlhs[int(interchangeRow1[posit]), j]
                # put newHypercube back in oldHypercube for the next iteration 
                outlhs = newHypercube.copy()
                # if this is not the first column we have used for this sweep 
                if j > 0:
                
                    # check to see how much benefit we gained from this sweep 
                    optimalityChange = np.abs(optimalityRecord[posit] - gOptimalityOld)
                    if optimalityChange < eps_change * optimalityChangeOld:
                        test = 1
                # if this is first column of the sweep, then store the benefit gained 
                else:
                
                    optimalityChangeOld = np.abs(optimalityRecord[posit] - gOptimalityOld)
                
                # replace the old optimality measure with the current one 
                gOptimalityOld = optimalityRecord[posit]
            
            # if the new and old optimality measures are equal 
            elif optimalityRecord[posit] == gOptimalityOld:
                test = 1
            
            # if the new optimality measure is worse 
            elif optimalityRecord[posit] > gOptimalityOld:
                raise Exception("Unexpected Result: Algorithm produced a less optimal design")
            if test == 1:
                break
            extraColumns += 1
    return outlhs




def optimumLHS(n, k, maxsweeps=2, eps=0.1):
    jLen = int(np.math.factorial(n)/(np.math.factorial(2.0)*np.math.factorial(n-2.0)) + 1.0)
    return np.random.rand(k)[np.newaxis, :] if n==0 else convertIntegerToNumericLhs(_optimumLHS(n, k, jLen, maxsweeps, eps))

