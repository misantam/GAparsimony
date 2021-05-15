import numpy as np
import pandas as pd
# # to_dict
# # =========================
# df.to_dict(orient='records')

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
                    

    def __setitem__(self, key, newvalue): # Tunear para slices
        newvalue = newvalue if len(newvalue.shape) > 1 else newvalue[np.newaxis, :]
        if type(key) is tuple:
            if type(key[1]) is int:
                if key[1] < len(self._params):
                    self._params[key[1], key[0]] = newvalue
                else:
                    self._params[key[1] - len(self._params), key[0]] = newvalue
            elif type(key[1]) is slice:
                if key[1].start is None:
                    if key[1].stop is None:
                        self._params[key[1], key[0]] = (newvalue[:, :len(self._params)] if "array" in type(newvalue).__name__ else newvalue)
                        self._cols[key[1], key[0]] = (newvalue[:, len(self._params):] if "array" in type(newvalue).__name__ else newvalue)
                    elif key[1].stop <= len(self._params):
                        self._params[key[1], key[0]] = (newvalue[key[0], :key[1].stop] if "array" in type(newvalue).__name__ else newvalue)
                    else:
                        self._params[key[1], key[0]] = (newvalue[:, :len(self._params)] if "array" in type(newvalue).__name__ else newvalue)
                        self._cols[:key[1].stop-len(self._params), key[0]] = (newvalue[:, len(self._params):] if "array" in type(newvalue).__name__ else newvalue)
                elif key[1].start <= len(self._params):
                    if key[1].stop is None:
                        self._params[key[1], key[0]] = (newvalue[:, key[1].start:len(self._params)] if "array" in type(newvalue).__name__ else newvalue)
                        self._cols[::key[1].step, key[0]] = (newvalue[:, len(self._params)-key[1].start:] if "array" in type(newvalue).__name__ else newvalue)
                    elif key[1].stop <= len(self._params):
                        self._params[key[1], key[0]] = (newvalue[key[0], key[1].start:key[1].stop] if "array" in type(newvalue).__name__ else newvalue)
                    else:
                        self._params[key[1].start::key[1].step, key[0]] = (newvalue[:, key[1].start:len(self._params)] if "array" in type(newvalue).__name__ else newvalue)
                        self._cols[:key[1].stop-len(self._params), key[0]] = (newvalue[:, len(self._params):] if "array" in type(newvalue).__name__ else newvalue)
                else:
                    if key[1].stop is None:
                        self._cols[::key[1].step, key[0]] = (newvalue[:, :] if "array" in type(newvalue).__name__ else newvalue)
                    else:
                        self._cols[:key[1].stop-len(self._params), key[0]] = (newvalue[:, :key[1].stop].T if "array" in type(newvalue).__name__ else newvalue)
        else:
            self._params[:, key] = (newvalue[:, :len(self._params)] if "array" in type(newvalue).__name__ else newvalue)
            self._cols[:, key] = (newvalue[:, len(self._params):] if "array" in type(newvalue).__name__ else newvalue)

    # def __setitem__(self, key, newvalue): # Tunear para slices
    #     if type(key) is tuple:
    #         if type(key[1]) is int:
    #             if key[1] < len(self._params):
    #                 self._params[key[1], key[0]] = newvalue
    #             else:
    #                 self._cols[key[1]-len(self._params), key[0]] = newvalue
    #         else:
    #             if key[1].start is None or key[1].start < len(self._params):
    #                 slices = list()
    #                 if key[1].stop is None or key[1].stop >= len(self._params):
    #                     slices.append((slice(key[1].start, len(self._params), key[1].step), key[0]), slice(key[1].start, len(self._params)))
    #                 else:
    #                     slices.append((slice(key[1].start, key[1].stop, key[1].step), key[0]), slice(key[1].start, key[1].stop))
    #             if key[1].stop is None or key[1].stop > len(self._params):
    #                 if key[1].start is None or key[1].start < len(self._params):
    #                     if key[1].stop is None:
    #                         slices.append((slice(0, key[1].stop, key[1].step), key[0]),)
    #                     else:
    #                         slices[1] = ((slice(0, key[1].stop-len(self._params), key[1].step), key[0]),)

    #                     slices[1] + (slice(len(self._params), key[1].stop) if key[1].start is None else slice(len(self._params)-key[1].start, key[1].stop))
    #                 else:
    #                     self._cols[slice(key[1].start-len(self._params), key[1].stop, key[1].step), key[0]] = newvalue[slice(key[1].start+len(self._params), key[1].stop)]
    #     else:
    #         self._params[:, key] = newvalue
    #         self._cols[:, key] = newvalue



  
    # def __setitem__(self, key, newvalue): # Tunear para slices
    #     if type(key) is tuple:
    #         if key[1] < len(self._params):
    #             self._params[key[1], key[0]] = newvalue
    #         else:
    #             self._cols[key[1], key[0]] = newvalue
    #     else:
    #         self._params[:, key] = newvalue[:len(self._params)]
    #         self._cols[:, key] = newvalue[len(self._params):]

    # def __getitem__(self, key):
    #     if type(key) is tuple:
    #         if key[1] < len(self._params):
    #             return self._params[key[1], key[0]]
    #         else:
    #             return self._cols[key[1]-len(self._params), key[0]]
    #     else:
    #         return np.concatenate([self._params[:, key], self._cols[:, key]]).ravel()
  
    # def __setitem__(self, key, newvalue):
    #     if type(key) is tuple:
    #         if key[1] < len(self._params):
    #             self._params[key[1], key[0]] = newvalue
    #         else:
    #             self._cols[key[1], key[0]] = newvalue
    #     else:
    #         self._params[:, key] = newvalue[:len(self._params)]
    #         self._cols[:, key] = newvalue[len(self._params):]

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

