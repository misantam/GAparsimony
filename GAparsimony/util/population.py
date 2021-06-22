# -*- coding: utf-8 -*-

import numpy as np

class Population:

    INTEGER = 0
    FLOAT = 1
    # STRING = 2
    CONSTANT = 3

    def __init__(self, params, columns, population = None):
        r"""
        This class is used to model the population of the chromosomes of the genetic algorithms. 
        Allow chromosomes to have int, float, and string values. 


        Parameters
        ----------
        params : dict
            It is a dictionary with the model's hyperparameters to be adjusted and the range of values to search for.
            
            .. code-block::

                {
                    "<< hyperparameter name >>": {
                        "range": [<< minimum value >>, << maximum value >>],
                        "type": GAparsimony.FLOAT/GAparsimony.INTEGER/GAparsimony.STRING
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
            List with parameter names.
        _params : dict
            Dict with the params values.
        _constnames : list of str
            List with constants names.
        _const : dict
            Dict with the constants values.
        colsnames : list of str
            List with the columns names.
        """
        
        if type(params) is not dict:
            raise Exception("params must be of type dict !!!")

        # self._min = np.array([(0 if params[x]["type"] is Population.STRING else params[x]["range"][0]) for x in params if params[x]["type"] is not Population.CONSTANT])
        # self._max = np.array([(len(params[x]["range"]) if params[x]["type"] is Population.STRING else params[x]["range"][1]) for x in params if params[x]["type"] is not Population.CONSTANT])

        self._min = np.array([params[x]["range"][0] for x in params if params[x]["type"] is not Population.CONSTANT])
        self._max = np.array([params[x]["range"][1] for x in params if params[x]["type"] is not Population.CONSTANT])

        self.paramsnames = [x for x in params if params[x]["type"] is not Population.CONSTANT]
        self._params = dict((x, params[x]) for x in params if params[x]["type"] is not Population.CONSTANT)
        self._constnames = [x for x in params if params[x]["type"] is Population.CONSTANT]
        self._const = [params[x]["value"] for x in params if params[x]["type"] is Population.CONSTANT]

        columns = (columns if type(columns) is list else columns.tolist()) if hasattr(columns, '__iter__') else [f"col_{i}" for i in range(columns)]
        self._min = np.concatenate((self._min, np.zeros(len(columns))), axis=0)
        self._max = np.concatenate((self._max, np.ones(len(columns))), axis=0)
        self.colsnames = columns

        if population is not None:
            if type(population) is not np.ndarray or len(population.shape) < 2:
                raise Exception("Popularion is not a numpy matrix")
            self.population = population

    @property
    def population(self):
        return self._pop

    @population.setter
    def population(self, population):

        self._pop = np.apply_along_axis(lambda x: x.astype(object), 1, population.astype(object))
        
        for i in range(self._pop.shape[1]):
            param = self._params[self.paramsnames[i]] if i < len(self.paramsnames) else None
            self._pop[:, i] = self._converValue(self._pop[:, i], param)

    def _converValue(self, value, param, axis=0):
        if "array" in type(value).__name__:
            if param is None or param["type"] is Population.INTEGER:
                return np.vectorize(lambda x: int(x), otypes=[int])(value)
            elif param["type"] is Population.FLOAT:
                return np.vectorize(lambda x: float(x), otypes=[float])(value)
            # elif param["type"] is Population.STRING:
            #     return np.vectorize(lambda x: x if type(x) is str else param["range"][int(np.trunc(x))], otypes=[str])(value)

        else:
            if param is None or param["type"] is Population.INTEGER:
                return int(value)
            elif param["type"] is Population.FLOAT:
                return float(value)
            # elif param["type"] is Population.STRING:
            #     return param["range"][int(np.trunc(value))]

    def __getitem__(self, key):
        return self._pop[key]

    def __setitem__(self, key, newvalue):
        if type(key) is tuple:
            if type(key[1]) is slice:
                start = 0 if key[1].start is None else key[1].start
                stop = len(self.paramsnames) if key[1].stop is None else key[1].stop
            else:
                start, stop = key[1], key[1]+1
        else:
            start, stop = 0, len(self.paramsnames)

        if "array" in type(newvalue).__name__:
            if len(newvalue.shape) > 1:
                newvalue = np.apply_along_axis(lambda x: x.astype(object), 1, newvalue.astype(object))
                for i in range(start, stop):
                    param = self._params[self.paramsnames[i]] if i < len(self.paramsnames) else None
                    newvalue[:, i] = self._converValue(newvalue[:, i], param)
                self._pop[key] = newvalue
            else:
                newvalue = newvalue.astype(object)
                for i in range(start, stop):
                    param = self._params[self.paramsnames[i]] if i < len(self.paramsnames) else None
                    newvalue[i] = self._converValue(newvalue[i], param)
                self._pop[key] = newvalue
        else:
            key = key[0] if type(key) is tuple else key
            for i in range(start, stop):
                param = self._params[self.paramsnames[i]] if i < len(self.paramsnames) else None
                self._pop[key, i] = self._converValue(newvalue, param)

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
        return Chromosome(self._pop[key, :len(self.paramsnames)], self.paramsnames, self._const, self._constnames, self._pop[key, len(self.paramsnames):], self.colsnames)

    
class Chromosome:

    # @autoassign
    def __init__(self, params, name_params, const, name_const, cols, name_cols):
        r"""
        This class models a chromosome, allowing the hyperparameters and column selection to be obtained in a simple way.


        Parameters
        ----------
        params : numpy.array
            The hyperparameters of the chromosome.
        name_params : list of str
            The names of the params.
        const : numpy.array
            The constants of the chromosome.
        name_const : list of str
            The names of the constants.
        cols : numpy.array
            The columns of the chromosome.
        name_cols : list of str
            The names of the columns.

        Attributes
        ----------
        params : dict
            A dictionary whose keys are the name of the parameter and its value the value of the parameter.
        columns : numpy.array of bool
            A boolean vector with the selected columns.
        """
        self._params = params.tolist()
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
        return self._cols>0.5