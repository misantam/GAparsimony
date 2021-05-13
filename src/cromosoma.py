import numpy as np


class Population:

    INTEGER = 0
    FLOAT = 1
    STRING = 2
    CONSTANT = 3

    def __init__(self, params, population, columns=None):
        
        if type(params) is not dict:
            raise Exception("La variable params tiene que ser de tipo dict!!!")

        if type(population) is not np.ndarray or len(population.shape) < 2:
            raise Exception("Popularion is not a numpy matrix")

        self.params_dict = params
        self.population = population


    @property
    def population(self):
        return np.concatenate([self._params, self._cols]).T

    @population.setter
    def population(self, population):
        self._min = [(0 if self.params_dict[x]["type"] is Population.STRING else self.params_dict[x]["range"][0]) for x in self.params_dict if self.params_dict[x]["type"] is not Population.CONSTANT]
        self._max = [(len(self.params_dict[x]["range"]) if self.params_dict[x]["type"] is Population.STRING else self.params_dict[x]["range"][1]) for x in self.params_dict if self.params_dict[x]["type"] is not Population.CONSTANT]

        population = population.T
        
        self._paramsnames = [x for x in self.params_dict if self.params_dict[x]["type"] is not Population.CONSTANT]
        self._params = list()
        for x in self.params_dict:
            if self.params_dict[x]["type"] is Population.INTEGER:
                self._params.append(population[len(self._params)].astype(np.uint8))
            elif self.params_dict[x]["type"] is Population.FLOAT:
                self._params.append(population[len(self._params)].astype(np.float32))
            elif self.params_dict[x]["type"] is Population.STRING:
                self._params.append(np.array(list(map(lambda p: self.params_dict[x]["range"][int(np.trunc(p))], population[len(self._params)]))))

        self._params = np.array(self._params, dtype=np.object)

        self._constnames = [x for x in self.params_dict if self.params_dict[x]["type"] is Population.CONSTANT]
        self._const = [self.params_dict[x]["value"] for x in self.params_dict if self.params_dict[x]["type"] is Population.CONSTANT]

        self._cols = population[len(self._paramsnames):]


    def __getitem__(self, key):
        if type(key) is tuple:
            if type(key[1]) is int:
                return self._params[key[1], key[0]] if key[1] < len(self._params) else self._cols[key[1]-len(self._params), key[0]]
            else:
                dev = list()
                if key[1].start is None or key[1].start < len(self._params):
                    if key[1].stop is None or key[1].stop >= len(self._params):
                        dev.append(self._params[slice(key[1].start, len(self._params), key[1].step), key[0]])
                    else:
                        dev.append(self._params[slice(key[1].start, key[1].stop, key[1].step), key[0]])
                if key[1].stop is None or key[1].stop > len(self._params):
                    if key[1].start is None:
                        dev.append(self._cols[slice(0, key[1].stop, key[1].step), key[0]])
                    else:
                        dev.append(self._cols[slice(key[1].start-len(self._params), key[1].stop, key[1].step), key[0]])
                return np.concatenate(dev).T
        else:
            return np.concatenate([self._params[:, key], self._cols[:, key]]).T if type(key) is slice else np.concatenate([self._params[:, key], self._cols[:, key]])
                    

    def __setitem__(self, key, newvalue):
        if type(key) is tuple:
            if type(key[1]) is int:
                if key[1] < len(self._params):
                    self._params[key[1], key[0]] = self._setValue(newvalue, (slice(1), slice(key[0], key[0]+1)), True)
                else:
                    self._params[key[1] - len(self._params), key[0]] = self._setValue(newvalue, (slice(1), slice(None)), False)
            elif type(key[1]) is slice:
                if key[1].start is None:
                    if key[1].stop is None:
                        self._params[key[1], key[0]] = self._setValue(newvalue, (key[0], slice(len(self._params))), True)
                        self._cols[key[1], key[0]] = self._setValue(newvalue, (key[0], slice(len(self._params), None)), False)
                    elif key[1].stop <= len(self._params):
                        self._params[key[1], key[0]] = self._setValue(newvalue, (key[0], slice(key[1].stop)), True)
                    else:
                        self._params[key[1], key[0]] = self._setValue(newvalue, (key[0], slice(len(self._params))), True)
                        self._cols[:key[1].stop-len(self._params), key[0]] = self._setValue(newvalue, (key[0], slice(len(self._params), None)), False)
                elif key[1].start <= len(self._params):
                    if key[1].stop is None:
                        self._params[key[1], key[0]] = self._setValue(newvalue, (key[0], slice(key[1].start, len(self._params))), True)
                        self._cols[::key[1].step, key[0]] = self._setValue(newvalue, (key[0], slice(len(self._params)-key[1].start, None)), False)
                    elif key[1].stop <= len(self._params):
                        self._params[key[1], key[0]] = self._setValue(newvalue, (key[0], slice(key[1].start, key[1].stop)), True)
                    else:
                        self._params[key[1].start::key[1].step, key[0]] = self._setValue(newvalue, (key[0], slice(key[0].start, len(self._params))), True)
                        self._cols[:key[1].stop-len(self._params), key[0]] = self._setValue(newvalue, (key[0], slice(len(self._params), None)), False)
                else:
                    if key[1].stop is None:
                        self._cols[::key[1].step, key[0]] = self._setValue(newvalue, (key[0], slice(None)), False)
                    else:
                        self._cols[:key[1].stop-len(self._params), key[0]] = self._setValue(newvalue, (key[0], slice(key[1].stop)), False)
        else:
            self._params[:, key] = self._setValue(newvalue, (key[0], slice(len(self._params))), True)
            self._cols[:, key] = self._setValue(newvalue, (key[0], slice(len(self._params), None)), False)


    def _setValue(self, value, slices, params):
        start_0 = 0 if slices[0].start is None else slices[0].start
        stop_0 = len(self._params) if slices[0].stop is None else slices[0].stop

        start_1 = 0 if slices[1].start is None else slices[1].start
            
        dev = []
        if params:
            stop_1 = len(self._params) if slices[1].stop is None else slices[1].stop
            if type(value) is int:
                    
                for x in range(start_1, stop_1):
                    if self.params_dict[self._paramsnames[x]]["type"] is Population.INTEGER:
                        dev.append(value.astype(np.uint8))
                    elif self.params_dict[self._paramsnames[x]]["type"] is Population.FLOAT:
                        dev.append(value.astype(np.float32))
                    elif self.params_dict[self._paramsnames[x]]["type"] is Population.STRING:
                        dev.append(self.params_dict[self._paramsnames[x]]["range"][int(np.trunc(value))])
                return (np.array(dev[0], dtype=np.object) if len(dev[0]) > 1 else dev[0][0]) if stop_0 is None or stop_0 < 2 else np.array(dev, dtype=np.object).T

            elif len(value.shape) == 1:

                value = value[np.newaxis] if len(value.shape) == 1 else value

                
                for v in value:
                    aux = list()
                    for x in range(start_1, stop_1):
                        if self.params_dict[self._paramsnames[x]]["type"] is Population.INTEGER:
                            aux.append(v[x].astype(np.uint8))
                        elif self.params_dict[self._paramsnames[x]]["type"] is Population.FLOAT:
                            aux.append(v[x].astype(np.float32))
                        elif self.params_dict[self._paramsnames[x]]["type"] is Population.STRING:
                            aux.append(self.params_dict[self._paramsnames[x]]["range"][int(np.trunc(v[x]))])
                    dev.append(aux)
                return (np.array(dev[0], dtype=np.object) if len(dev[0]) > 1 else dev[0][0]) if stop_0 is None or stop_0 < 2 else np.array(dev, dtype=np.object).T
                    
            else:

                
                for y in range(start_0, stop_0):
                    aux = list()
                    for x in range(start_1, stop_1):
                        if self.params_dict[self._paramsnames[x]]["type"] is Population.INTEGER:
                            aux.append(value[y][x].astype(np.uint8))
                        elif self.params_dict[self._paramsnames[x]]["type"] is Population.FLOAT:
                            aux.append(value[y][x].astype(np.float32))
                        elif self.params_dict[self._paramsnames[x]]["type"] is Population.STRING:
                            aux.append(self.params_dict[self._paramsnames[x]]["range"][int(np.trunc(value[y][x]))])
                    dev.append(aux)
                return (np.array(dev[0], dtype=np.object) if len(dev[0]) > 1 else dev[0][0]) if stop_0 is None or stop_0 < 2 else np.array(dev, dtype=np.object).T
        
        else:
            

            if type(value) is int:
                return int(value)
            elif len(value.shape) == 1:
                stop_1 = len(value) if slices[1].stop is None else slices[1].stop
                return value[start_1:stop_1].astype(np.uint8) if stop_0 is None or stop_0 < 2 else value[np.newaxis, start_1:stop_1].astype(np.uint8).T
            else:
                stop_1 = len(value[0]) if slices[1].stop is None else slices[1].stop
                return value[:, start_1:stop_1].astype(np.uint8).T

    def getCromosoma(self, key):
        return Cromosoma(self._params[:, key], self._paramsnames, self._const, self._constnames, self._cols, None)

    
    
class Cromosoma:

    # @autoassign
    def __init__(self, params, name_params, const, name_const, cols, name_cols):
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
        return self._cols