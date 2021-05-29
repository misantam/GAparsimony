import numpy as np
from src.gaparsimony import GAparsimony
from src.population import Population

from test.utilTest import autoargs, readJSONFile


# Clase generica, permite instanciarla con diferentes atributos.
class GenericClass(object):
    @autoargs()
    def __init__(self,**kawargs):
        pass

data = readJSONFile('./test/outputs/populationClass2.json')
population = Population(data["params"], data["features"], np.array(data["population"]))

# @pytest.mark.parametrize("population, slice, value, resultado", 
#                         [(population,(slice(2), slice(None)), np.arange(20), np.array(data["population_1"], dtype=object)),
#                         (population,(slice(2), slice(None)), np.array([np.arange(20), np.arange(1, 21)]), np.array(data["population_2"], dtype=object)),
#                         (population,(slice(2), slice(None)), 0, np.array(data["population_3"], dtype=object)),
#                         (population,(1, slice(2)), 1, np.array(data["population_4"], dtype=object)),
#                         (population,(1, slice(None)), np.arange(20), np.array(data["population_5"], dtype=object)),
#                         (population,(1, slice(2)), np.array([2,2]), np.array(data["population_6"], dtype=object)),
#                         (population,(slice(None), 2), 1, np.array(data["population_7"], dtype=object)),
#                         (population,(slice(None), 6), 87, np.array(data["population_8"], dtype=object)),
#                         (population,(slice(None), 7), 98, np.array(data["population_9"], dtype=object))])
def test_GAParsimony_regresion_population_class(population, slice, value, resultado):

    population[slice] = value
    
    assert (population.population==resultado).all()


test_GAParsimony_regresion_population_class(population,(slice(2), slice(None)), np.arange(20), np.array(data["population_1"], dtype=object))