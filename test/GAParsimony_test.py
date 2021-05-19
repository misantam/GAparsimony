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
def test_GAParsimony_regresion_boston_population(population):

    min_param = np.concatenate((np.array([1., 0.0001]), np.zeros(13)), axis=0)
    max_param = np.concatenate((np.array([25, 0.9999]), np.ones(13)), axis=0)
    
    model = GenericClass(nParams=2, nFeatures=13, popSize=40, seed_ini=1234, max_param=max_param, min_param=min_param, feat_thres=0.90, population=None)

    GAparsimony._population(model, type_ini_pop="improvedLHS")
    
    assert (model.population==population).all()


with open('./test/outputs/rerank.json') as f:
    rerank = json.load(f)

@pytest.mark.parametrize("rerank", [rerank])
def test_GAParsimony_regresion_boston_rerank(rerank):

    
    model = GenericClass(fitnessval=np.array(rerank["fitnessval"]), complexity=np.array(rerank["complexity"]), 
                        best_score=rerank["best_score"], popSize=rerank["popSize"], rerank_error = 0.01, verbose=0)

    result = GAparsimony._rerank(model)
    
    assert (result==np.array(rerank["position"])).all()


with open('./test/outputs/selection.json') as f:
    selection = json.load(f)

@pytest.mark.parametrize("selection", [selection])
def test_GAParsimony_regresion_boston_selection(selection):
    np.random.seed(selection["seed"])
    population=np.array(selection["population"])
    fitnessval=np.array(selection["fitnessval"])
    fitnesstst=np.array(selection["fitnesstst"])
    complexity=np.array(selection["complexity"])
    
    model = GenericClass(selection=selection["selection"], popSize=selection["popSize"], 
                        sel=selection["sel"], population=population, fitnessval=fitnessval,
                        complexity=complexity, fitnesstst=fitnesstst
                        )

    GAparsimony._selection(model)
    
    assert (model.population==population[selection["sel"]]).all() and (model.fitnessval==fitnessval[selection["sel"]]).all() and \
            (model.fitnesstst==fitnesstst[selection["sel"]]).all() and (model.complexity==complexity[selection["sel"]]).all()

# @pytest.mark.parametrize("shape", [
#     (2, 2),
#     (6, 6),
#     (3, 8)
# ])
# def test_randomLHS(shape):
#     assert isValidLHS(randomLHS(*shape))

