# -*- coding: utf-8 -*-

import numpy as np
from GAparsimony import GAparsimony, Population

import pytest
from .utilTest import autoargs, readJSONFile


# Clase generica, permite instanciarla con diferentes atributos.
class GenericClass(object):
    @autoargs()
    def __init__(self,**kawargs):
        pass


#################################################
#***************TEST POPULATION*****************#
#################################################

@pytest.mark.parametrize("population", [(readJSONFile('./test/outputs/population.json'))])
def test_GAParsimony_regression_boston_population(population):

    pop = Population(population["params"], columns=population["features"])
    
    model = GenericClass(popSize=population["popSize"], seed_ini=population["seed"], feat_thres=population["feat_thres"], population=pop)

    pop.population = GAparsimony._population(model, type_ini_pop="improvedLHS")
    
    assert (pop.population==np.array(population["population_resultado"])).all()


data = readJSONFile('./test/outputs/populationClass.json')
population = Population(data["params"], data["features"], np.array(data["population"]))

@pytest.mark.parametrize("population, slice, value, resultado", 
                        [(population,(slice(2), slice(None)), np.arange(20), np.array(data["population_1"], dtype=object)),
                        (population,(slice(2), slice(None)), np.array([np.arange(20), np.arange(1, 21)]), np.array(data["population_2"], dtype=object)),
                        (population,(slice(2), slice(None)), 0, np.array(data["population_3"], dtype=object)),
                        (population,(1, slice(2)), 1, np.array(data["population_4"], dtype=object)),
                        (population,(1, slice(None)), np.arange(20), np.array(data["population_5"], dtype=object)),
                        (population,(1, slice(2)), np.array([2,2]), np.array(data["population_6"], dtype=object)),
                        (population,(slice(None), 2), 1, np.array(data["population_7"], dtype=object)),
                        (population,(slice(None), 6), 87, np.array(data["population_8"], dtype=object)),
                        (population,(slice(None), 7), 98, np.array(data["population_9"], dtype=object))])
def test_GAParsimony_regression_population_class(population, slice, value, resultado):

    population[slice] = value
    
    assert (population.population==resultado).all()

#################################################
#*****************TEST RERANK*******************#
#################################################

@pytest.mark.parametrize("rerank", [readJSONFile('./test/outputs/rerank.json')])
def test_GAParsimony_regression_boston_rerank(rerank):

    
    model = GenericClass(fitnessval=np.array(rerank["fitnessval"]), complexity=np.array(rerank["complexity"]), 
                        best_score=rerank["best_score"], popSize=rerank["popSize"], rerank_error = 0.01, verbose=0, 
                        population=Population(rerank["params"], rerank["popSize"]))

    result = GAparsimony._rerank(model)
    
    assert (result==np.array(rerank["position"])).all()


#################################################
#****************TEST SELECTION*****************#
#################################################

@pytest.mark.parametrize("selection", [readJSONFile('./test/outputs/selection.json')])
def test_GAParsimony_regression_boston_selection(selection):
    np.random.seed(selection["seed"])
    population=np.array(selection["population"])
    fitnessval=np.array(selection["fitnessval"])
    fitnesstst=np.array(selection["fitnesstst"])
    complexity=np.array(selection["complexity"])
    
    model = GenericClass(selection=selection["selection"], popSize=selection["popSize"], 
                        sel=selection["sel"], population=Population(selection["params"], selection["features"], np.array(selection["population"])), fitnessval=fitnessval,
                        complexity=complexity, fitnesstst=fitnesstst
                        )

    GAparsimony._selection(model)
    
    assert (model.population.population==population[selection["sel"]]).all() and (model.fitnessval==fitnessval[selection["sel"]]).all() and \
            (model.fitnesstst==fitnesstst[selection["sel"]]).all() and (model.complexity==complexity[selection["sel"]]).all()

#################################################
#****************TEST MUTATION*****************#
#################################################

@pytest.mark.parametrize("mutation", [readJSONFile('./test/outputs/mutation.json')])
def test_GAParsimony_regression_boston_mutation(mutation):
    np.random.seed(mutation["seed"])
    
    model = GenericClass(pmutation=mutation["pmutation"], popSize=mutation["popSize"], not_muted=mutation["not_muted"], 
                        population=Population(mutation["params"], mutation["features"], np.array(mutation["population"])), 
                        feat_mut_thres=mutation["feat_mut_thres"],
                        fitnessval=np.array(mutation["fitnessval"]), fitnesstst=np.array(mutation["fitnesstst"]), complexity=np.array(mutation["complexity"]))

    GAparsimony._mutation(model)
    
    assert (model.population.population==np.array(mutation["resultado"])).all()

#################################################
#****************TEST CROSSOVER*****************#
#################################################

@pytest.mark.parametrize("crossover", [readJSONFile('./test/outputs/crossover.json')])
def test_GAParsimony_regression_boston_crossover(crossover):
    np.random.seed(crossover["seed"])
    
    model = GenericClass(pcrossover=crossover["pcrossover"],
                        popSize=crossover["popSize"], population=Population(crossover["params"], crossover["features"], np.array(crossover["population"])), 
                        fitnessval=np.array(crossover["fitnessval"]), 
                        fitnesstst=np.array(crossover["fitnesstst"]), complexity=np.array(crossover["complexity"]))

    nmating = int(np.floor(model.popSize/2))
    mating = np.random.choice(list(range(2 * nmating)), size=(2 * nmating), replace=False).reshape((nmating, 2))
    for i in range(nmating):
        if model.pcrossover > np.random.uniform(low=0, high=1):
            parents = mating[i, ]
            GAparsimony._crossover(model, parents=parents)
    
    assert (model.population.population==np.array(crossover["resultado"])).all()
