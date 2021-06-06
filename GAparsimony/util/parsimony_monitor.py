# -*- coding: utf-8 -*-

import numpy  as np


# new function for monitoring
def parsimony_monitor(object, digits = 7, *args):
    r"""Functions for monitoring GA-PARSIMONY algorithm evolution

    Functions to print summary statistics of fitness values at each iteration of a GA search.

    Parameters
    ----------
    object : object of GAparsimony
        The `GAparsimony` object that we want to monitor .
    digits : int
        Minimal number of significant digits.
    *args : 
        Further arguments passed to or from other methods.
    """

    fitnessval = object.fitnessval[~np.isnan(object.fitnessval)]
        
    print(f"GA-PARSIMONY | iter = {object.iter}")
    print("|".join([f"MeanVal = {round(np.mean(fitnessval), digits)}".center(16+digits), 
                    f"ValBest = {round(object.bestfitnessVal, digits)}".center(16+digits),
                    f"TstBest = {round(object.bestfitnessTst, digits)}".center(16+digits),
                    f"ComplexBest = {round(object.bestcomplexity, digits)}".center(19+digits),
                    f"Time(min) = {round(object.minutes_gen, digits)}".center(17+digits)]) + "\n")
 



# Duda si es todo el rato con x1
# Equivalencia a fivenum es np.percentile(aux, [0, 25, 50, 75, 100])

def parsimony_summary(object, *args):
  
    x1 = object.fitnessval[~np.isnan(object.fitnessval)]
    q1 = np.percentile(x1, [0, 25, 50, 75, 100])
    x2 = object.fitnesstst[~np.isnan(object.fitnesstst)]
    q2 = np.percentile(x1, [0, 25, 50, 75, 100])
    x3 = object.complexity[~np.isnan(object.complexity)]
    q3 = np.percentile(x1, [0, 25, 50, 75, 100])

    return q1[4], np.mean(x1), q1[3], q1[2], q1[1], q1[0], q2[4], np.mean(x2), q2[3], q2[2], q2[1], q2[0], q3[4], np.mean(x3), q3[3], q3[2], q3[1], q3[0]



