# -*- coding: utf-8 -*-

import os
import numpy as np


# #----------------------------------------------------------------------------#
# # print a short version of a matrix by allowing to select the number of 
# # head/tail rows and columns to display

def printShortMatrix(x, head=2, tail = 1, chead = 5, ctail = 1, **kwargs):
  # x es una matriz
  nr, nc = x.shape

  
  if nr > (head + tail + 1):
    # rnames = rnames if "rnames" in kwargs else print(range(nr).join(", "))
    x = np.stack([x[:head, :], np.repeat(np.nan, nc), x[(nr-tail+1):, :]], axis=0)
    # rownames(x) <- c(rnames[1:head], "...", rnames[(nr-tail+1):nr])
  if nc > (chead + ctail + 1):
    # cnames <- colnames(x)
    if not "cnames" in kwargs:
      # cnames = range(nr).join(", ")
      # x <- cbind(x[,1:chead,drop=FALSE], rep(NA, nrow(x)), x[,(nc-ctail+1):nc,drop=FALSE])
      x = np.stack([x[:, :chead], np.repeat(np.nan, nr), x[:, (nc-ctail+1):nc]], axis=0)
      # colnames(x) <- c(cnames[1:chead], "...", cnames[(nc-ctail+1):nc])

  print(x)
