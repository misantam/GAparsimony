# -*- coding: utf-8 -*-

import numpy as np

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