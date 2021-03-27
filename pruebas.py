import numpy as np

# #               ##
# #               ##
# # GENETIC LHS   ##
# #               ##
# #               ##

from lhs.base.geneticLHS import geneticLHS
from lhs.util.utilityLHS import isValidLHS, isValidLHS_int


m =geneticLHS(6, 6)

print(isValidLHS(m))


# #               ##
# #               ##
# # OPTIMUM LHS   ##
# #               ##
# #               ##

from lhs.base.optimumLHS import optimumLHS
from lhs.util.utilityLHS import isValidLHS, isValidLHS_int


m =optimumLHS(5,5)

print(isValidLHS(m))

