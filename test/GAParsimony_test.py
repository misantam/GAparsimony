import numpy as np
from src.gaparsimony import GAparsimony

import pytest, json
from .utilTest import autoargs

class GenericClass(object):
    @autoargs()
    def __init__(self,**kawargs):
        pass


#################################################
#***************TEST POPULATION*****************#
#################################################

with open('./test/outputs/population.json') as f:
    population = np.array(json.load(f))

@pytest.mark.parametrize("population", [(population)])
def test_GAParsimony_regresion_boston_population(population):

    min_param = np.concatenate((np.array([1., 0.0001]), np.zeros(13)), axis=0)
    max_param = np.concatenate((np.array([25, 0.9999]), np.ones(13)), axis=0)
    
    model = GenericClass(nParams=2, nFeatures=13, popSize=40, seed_ini=1234, max_param=max_param, min_param=min_param, feat_thres=0.90, population=None)

    GAparsimony._population(model, type_ini_pop="improvedLHS")
    
    assert (model.population==population).all()



#################################################
#*****************TEST RERANK*******************#
#################################################

with open('./test/outputs/rerank.json') as f:
    rerank = json.load(f)

@pytest.mark.parametrize("rerank", [rerank])
def test_GAParsimony_regresion_boston_rerank(rerank):

    
    model = GenericClass(fitnessval=np.array(rerank["fitnessval"]), complexity=np.array(rerank["complexity"]), 
                        best_score=rerank["best_score"], popSize=rerank["popSize"], rerank_error = 0.01, verbose=0)

    result = GAparsimony._rerank(model)
    
    assert (result==np.array(rerank["position"])).all()


#################################################
#****************TEST SELECTION*****************#
#################################################

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


with open('./test/outputs/mutation.json') as f:
    mutation = json.load(f)

@pytest.mark.parametrize("mutation", [mutation])
def test_GAParsimony_regresion_boston_mutation(mutation):
    np.random.seed(mutation["seed"])
    
    model = GenericClass(pmutation=mutation["pmutation"], nParams=mutation["nParams"], nFeatures=mutation["nFeatures"],
                        popSize=mutation["popSize"], not_muted=mutation["not_muted"], population=np.array(mutation["population"]), 
                        min_param=np.array(mutation["min_param"]), max_param=np.array(mutation["max_param"]), feat_mut_thres=mutation["feat_mut_thres"],
                        fitnessval=np.array(mutation["fitnessval"]), fitnesstst=np.array(mutation["fitnesstst"]), complexity=np.array(mutation["complexity"]))

    GAparsimony._mutation(model)
    
    assert (model.population==np.array(mutation["resultado"])).all()

#################################################
#****************TEST SELECTION*****************#
#################################################

with open('./test/outputs/crossover.json') as f:
    crossover = json.load(f)

@pytest.mark.parametrize("crossover", [crossover])
def test_GAParsimony_regresion_boston_crossover(crossover):
    np.random.seed(crossover["seed"])
    
    model = GenericClass(nFeatures=crossover["nFeatures"], nParams=crossover["nParams"], pcrossover=crossover["pcrossover"],
                        popSize=crossover["popSize"], population=np.array(crossover["population"]), min_param=np.array(crossover["min_param"]), 
                        max_param=np.array(crossover["max_param"]), fitnessval=np.array(crossover["fitnessval"]), 
                        fitnesstst=np.array(crossover["fitnesstst"]), complexity=np.array(crossover["complexity"]))

    nmating = int(np.floor(model.popSize/2))
    mating = np.random.choice(list(range(2 * nmating)), size=(2 * nmating), replace=False).reshape((nmating, 2))
    for i in range(nmating):
        if model.pcrossover > np.random.uniform(low=0, high=1):
            parents = mating[i, ]
            GAparsimony._crossover(model, parents=parents)
    
    assert (model.population==np.array(crossover["resultado"])).all()
