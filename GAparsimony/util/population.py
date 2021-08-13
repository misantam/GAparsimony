# -*- coding: utf-8 -*-

import numpy as np

class Population:

    INTEGER = 0
    FLOAT = 1
    STRING = 2
    CONSTANT = 3

    def __init__(self, params, columns, population = None):
        r"""
        This class is used to create the GA populations. 
        Allow chromosomes to have int, float, and constant values. 


        Parameters
        ----------
        params : dict
            It is a dictionary with the model's hyperparameters to be adjusted and the search space of them.
            
            .. code-block::

                {
                    "<< hyperparameter name >>": {
                        "range": [<< minimum value >>, << maximum value >>],
                        "type": GAparsimony.FLOAT/GAparsimony.INTEGER
                    },
                    "<< hyperparameter name >>": {
                        "value": << constant value >>,
                        "type": GAparsimony.CONSTANT
                    }
                }

        columns : int or list of str
            The number of features/columns in the dataset or a list with their names.
        population : numpy.array, optional
            It is a float matrix that represents the population. Default `None`.

        Attributes
        ----------
        population : Population
            The population.
        _min : numpy.array
            A vector of length `params+columns` with the smallest values that can take.
        _max : numpy.array
            A vector of length `params+columns` with the highest values that can take.
        paramsnames : list of str
            List with the parameter names.
        _params : dict
            Dict with the parameter values.
        _constnames : list of str
            List with the constants names.
        _const : dict
            Dict with the constants values.
        colsnames : list of str
            List with the columns names.
        """
        
        if type(params) is not dict:
            raise Exception("params must be of type dict !!!")

        self._min = np.array([(0 if params[x]["type"] is Population.STRING else params[x]["range"][0]) for x in params if params[x]["type"] is not Population.CONSTANT])
        self._max = np.array([(len(params[x]["range"]) if params[x]["type"] is Population.STRING else params[x]["range"][1]) for x in params if params[x]["type"] is not Population.CONSTANT])

        self.paramsnames = [x for x in params if params[x]["type"] is not Population.CONSTANT]
        self._params = dict((x, params[x]) for x in params if params[x]["type"] is not Population.CONSTANT)
        self._constnames = [x for x in params if params[x]["type"] is Population.CONSTANT]
        self._const = [params[x]["value"] for x in params if params[x]["type"] is Population.CONSTANT]

        columns = (columns if type(columns) is list else columns.tolist()) if hasattr(columns, '__iter__') else [f"col_{i}" for i in range(columns)]
        self._min = np.concatenate((self._min, np.zeros(len(columns))), axis=0)
        self._max = np.concatenate((self._max, np.ones(len(columns))), axis=0)
        self.colsnames = columns

        def _trans():
            t = list()
            for x in self.paramsnames:
                if params[x]["type"] == Population.INTEGER:
                    t.append(np.vectorize(lambda x: int(x), otypes=[int]))
                elif params[x]["type"] == Population.FLOAT:
                    t.append(np.vectorize(lambda x: float(x), otypes=[float]))
                elif params[x]["type"] == Population.STRING:
                    t.append(np.vectorize(lambda y, x=x: y if type(y) is str else params[x]["range"][int(np.trunc(y))], otypes=[str]))
            t.extend([lambda x: x>0.5]*len(self.colsnames))

            def aux(x):
                if len(x.shape)>1:
                    return np.array(list(map(lambda f, c: f(x[:, c]), t, range(0, x.shape[1]))), dtype=object).T
                else:
                    return list(map(lambda f, c: f(c), t, x))

            return aux


        self._transformers = _trans()
        

        if population is not None:
            if type(population) is not np.ndarray or len(population.shape) < 2:
                raise Exception("Popularion is not a numpy matrix")
            self.population = population

    @property
    def population(self):
        return self._transformers(self._pop)

    @population.setter
    def population(self, population):
        self._pop = np.apply_along_axis(lambda x: x.astype(object), 1, population.astype(object))

    def __getitem__(self, key):
        return self._pop[key]

    def __setitem__(self, key, newvalue):
        self._pop[key] = newvalue

    def getChromosome(self, key):
        r"""
        This method returns a chromosome from the population. 

        Parameters
        ----------
        key : int
            Chromosome row index .

        Returns
        -------
        Chromosome
            A `Chromosome` object.
        """
        data = self._transformers(self._pop[key, :])
        return Chromosome(data[:len(self.paramsnames)], self.paramsnames, self._const, self._constnames, data[len(self.paramsnames):], self.colsnames)

    
class Chromosome:

    # @autoassign
    def __init__(self, params, name_params, const, name_const, cols, name_cols):
        r"""
        This class defines a chromosome which includes the hyperparameters, the constant values, and the feature selection.


        Parameters
        ----------
        params : numpy.array
            The algorithm hyperparameter values.
        name_params : list of str
            The names of the hyperparameters.
        const : numpy.array
            The constants to include in the chomosome.
        name_const : list of str
            The names of the constants.
        cols : numpy.array
            The probabilities for selecting the input features (selected if prob>0.5).
        name_cols : list of str
            The names of the input features.

        Attributes
        ----------
        params : dict
            A dictionary with the parameter values (hyperparameters and constants).
        columns : numpy.array of bool
            A boolean vector with the selected features.
        """
        self._params = params
        self.name_params = name_params
        self.const = const
        self.name_const = name_const
        self._cols = cols
        self.name_cols = name_cols

    @property
    def params(self):
        return dict((x, y) for x, y in zip(self.name_params+self.name_const, self._params + self.const))

    @property
    def columns(self):
        return self._cols
