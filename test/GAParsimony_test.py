import numpy as np
from src.gaparsimony import GAparsimony

import pytest, json
from .utilTest import autoargs

class GenericClass(object):
    @autoargs()
    def __init__(self,**kawargs):
        pass



with open('./test/outputs/population.json') as f:
    population = np.array(json.load(f))

@pytest.mark.parametrize("population", [(population)])
def test_GAParsimony_regresion_boston(population):

    min_param = np.concatenate((np.array([1., 0.0001]), np.zeros(13)), axis=0)
    max_param = np.concatenate((np.array([25, 0.9999]), np.ones(13)), axis=0)
    
    model = GenericClass(nParams=2, nFeatures=13, popSize=40, seed_ini=1234, max_param=max_param, min_param=min_param, feat_thres=0.90, population=None)

    GAparsimony._population(model, type_ini_pop="improvedLHS")
    
    assert (model.population==population).all()

# @pytest.mark.parametrize("shape", [
#     (2, 2),
#     (6, 6),
#     (3, 8)
# ])
# def test_randomLHS(shape):
#     assert isValidLHS(randomLHS(*shape))

