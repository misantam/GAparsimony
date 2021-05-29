import numpy as np

class Population:

    INTEGER = 0
    FLOAT = 1
    STRING = 2
    CONSTANT = 3

    def __init__(self, params, columns, population = None):
        
        if type(params) is not dict:
            raise Exception("La variable params tiene que ser de tipo dict!!!")

        self._min = np.array([(0 if params[x]["type"] is Population.STRING else params[x]["range"][0]) for x in params if params[x]["type"] is not Population.CONSTANT])
        self._max = np.array([(len(params[x]["range"]) if params[x]["type"] is Population.STRING else params[x]["range"][1]) for x in params if params[x]["type"] is not Population.CONSTANT])

        self.paramsnames = [x for x in params if params[x]["type"] is not Population.CONSTANT]
        self._params = dict((x, params[x]) for x in params if params[x]["type"] is not Population.CONSTANT)
        self._constnames = [x for x in params if params[x]["type"] is Population.CONSTANT]
        self._const = [params[x]["value"] for x in params if params[x]["type"] is Population.CONSTANT]

        columns = columns if type(columns) is list else [f"col_{i}" for i in range(columns)]
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
            elif param["type"] is Population.STRING:
                return np.vectorize(lambda x: param["range"][int(np.trunc(x))], otypes=[str])(value)

        else:
            if param is None or param["type"] is Population.INTEGER:
                return int(value)
            elif param["type"] is Population.FLOAT:
                return float(value)
            elif param["type"] is Population.STRING:
                return param["range"][int(np.trunc(value))]

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

    def getCromosoma(self, key):
        return Cromosoma(self._pop[key, :len(self.paramsnames)], self.paramsnames, self._const, self._constnames, self._pop[key, len(self.paramsnames):], self.colsnames)

    
class Cromosoma:

    # @autoassign
    def __init__(self, params, name_params, const, name_const, cols, name_cols):
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