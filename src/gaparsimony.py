from src.population import Population
from .parsimony_monitor import parsimony_monitor, parsimony_summary
from .ordenacion import order
from lhs.base import *
from .parsimony_miscfun import printShortMatrix

import warnings
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import time
import inspect, opcode




class GAparsimony(object):

    MONITOR = 1
    DEBUG = 2

    def __init__(self, 
                fitness,
                params,
                features,
                type_ini_pop="improvedLHS", 
                popSize = 50, 
                pcrossover = 0.8,  
                maxiter = 40, 
                feat_thres=0.90, 
                rerank_error = 0.0, 
                iter_start_rerank = 0,
                pmutation = 0.10, 
                feat_mut_thres=0.10, 
                not_muted=3,
                elitism = None,
                selection = "nlinear", 
                keep_history = False,
                early_stop = None, 
                maxFitness = np.Inf, 
                suggestions = None,
                seed_ini = None, 
                verbose=MONITOR,
                logger = None):

        
        self.elitism = max(1, round(popSize * 0.20)) if not elitism else elitism

        self.population = Population(params, columns=features)


        # Check parameters
        # ----------------
        if selection is None or type(selection) is not str:
            raise ValueError("A selection(sting) must be provided!!!")
        if not fitness:
            raise Exception("A fitness function must be provided!!!")
        if not callable(fitness):
            raise Exception("A fitness function must be provided!!!")
        if popSize < 10:
            warnings.warn("The population size is less than 10!!!")
        if maxiter < 1:
            raise ValueError("The maximum number of iterations must be at least 1!!!")
        if self.elitism > popSize:
            raise ValueError("The elitism cannot be larger that population size.")
        if pcrossover < 0 or pcrossover > 1:
            raise ValueError("Probability of crossover must be between 0 and 1!!!")
        if pmutation < 0 or pmutation > 1:
            raise ValueError("Probability of mutation must be between 0 and 1!!!")
        if self.population._min is None and self.population._max is None:
            raise ValueError("A min and max range of values must be provided!!!")
        if self.population._min.shape != self.population._max.shape:
            raise Exception("min_param and max_param must have the same length!!!")
        if not features:
            raise Exception("Number of features or name of features must be provided!!!")
        if (suggestions is not None) or (type(suggestions) is list and len(suggestions)>0 and type(suggestions[0]) is not list) or (type(suggestions) is np.array and len(suggestions.shape) < 2):
            raise Exception("Provided suggestions is a vector")
        if (type(suggestions) is np.array) and (self.population._min.shape + features) != suggestions.shape[1]:
            raise Exception("Provided suggestions (ncol) matrix do not match the number of variables (model parameters + vector with selected features) in the problem!")

        self.call = locals()
        self.fitness = fitness
        self.popSize = popSize
        self.pcrossover = pcrossover
        self.maxiter = maxiter
        self.feat_thres=feat_thres
        self.rerank_error=rerank_error
        self.iter_start_rerank=iter_start_rerank
        self.pmutation = pmutation
        self.feat_mut_thres=feat_mut_thres
        self.not_muted=not_muted
        
        self.selection = selection
        self.keep_history = keep_history
        self.early_stop = maxiter if not early_stop else early_stop
        self.maxFitness = maxFitness
        self.suggestions = np.array(suggestions) if type(suggestions) is list else suggestions
        self.seed_ini = seed_ini
        self.verbose = verbose
        self.logger = None

        self.nvars = len(self.population.paramsnames) + (len(features) if type(features) is list else features)

        self.iter = 0
        self.minutes_total=0
        self.best_score = np.NINF
        self.history = list()

        if self.seed_ini:
            np.random.seed(self.seed_ini)

        self.population.population = self._population(type_ini_pop=type_ini_pop) # Creo la poblacion de la primera generacion

        if self.suggestions:
            ng = min(self.suggestions.shape[0], popSize)
            if ng > 0:
                self.population[:ng, :] = self.suggestions[:ng, :]

        
        
        
    def fit(self, iter_ini=0):
           
        # Get suggestions
        # ---------------
        if self.verbose == GAparsimony.DEBUG and self.suggestions:
            print(self.suggestions)


        # Initial settings
        # ----------------
        if self.seed_ini:
            np.random.seed(self.seed_ini)

        self._summary = np.empty((self.maxiter,6*3,))
        self._summary[:] = np.nan
        self.bestSolList = list()
        self.fitnessval = np.empty(self.popSize)
        self.fitnessval[:] = np.nan
        self.fitnesstst = np.empty(self.popSize)
        self.fitnesstst[:] = np.nan
        self.complexity = np.empty(self.popSize)
        self.complexity[:] = np.nan


        if len(self.history) > 0:
            if self.verbose == GAparsimony.DEBUG:
                print("There is a GAparsimony 'object'!!!")
                print(self)

            iter_ini = self.iter if not iter_ini else min(iter_ini, self.iter)
            if iter_ini < 0:
                iter_ini = 0

            self.history = self.history[iter_ini].values[0]
            if self.verbose == GAparsimony.DEBUG:
                print(f"Starting GA optimization with a provided GAparsimony 'object'. Using object's GA settings and its population from iter={iter_ini}.")
        
        elif self.verbose == GAparsimony.DEBUG:
            print("\nStep 0. Initial population")
            print(np.c_[self.fitnessval, self.fitnesstst, self.complexity, self.population.population][:10, :])
            # input("Press [enter] to continue")


        # Main Loop
        # --------- 
        for iter in range(self.maxiter):
            tic = time.time()
            
            self.iter = iter

            for t in range(self.popSize):
                if np.isnan(self.fitnessval[t]) and np.sum(self.population[t,range(len(self.population.paramsnames), self.nvars)])>0:
                    fit = self.fitness(self.population[t])
                    self.fitnessval[t] = fit[0]
                    self.fitnesstst[t] = fit[1]
                    self.complexity[t] = fit[2]
                
            
            # np.random.seed(self.seed_ini*iter) if not self.seed_ini else np.random.seed(1234*iter)
            if self.seed_ini:
                np.random.seed(self.seed_ini*iter) 
                # np.random.seed(self.seed_ini) 
            

            # Sort by the Fitness Value
            # ----------------------------
            ord = order(self.fitnessval, kind='heapsort', decreasing = True, na_last = True)
            self.population.population = self.population[ord, :]
            self.fitnessval = self.fitnessval[ord]
            self.fitnesstst = self.fitnesstst[ord]
            self.complexity = self.complexity[ord]

            PopSorted = self.population.population.copy()
            FitnessValSorted = self.fitnessval.copy()
            FitnessTstSorted = self.fitnesstst.copy()
            ComplexitySorted = self.complexity.copy()
            
            if np.max(self.fitnessval)>self.best_score:
                self.best_score = np.nanmax(self.fitnessval)
                self.solution_best_score = np.r_[self.best_score, 
                                                self.fitnesstst[np.argmax(self.fitnessval)], 
                                                self.complexity[np.argmax(self.fitnessval)], 
                                                self.population[np.argmax(self.fitnessval)]]


            if self.verbose == GAparsimony.DEBUG:
                print("\nStep 1. Fitness sorted")
                print(np.c_[self.fitnessval, self.fitnesstst, self.complexity, self.population.population][:10, :])
                # input("Press [enter] to continue")

            
            # Reorder models with ReRank function
            # -----------------------------------
            if self.rerank_error != 0.0 and self.iter >= self.iter_start_rerank:
                ord_rerank = self._rerank()
                self.population.population = self.population[ord_rerank]
                self.fitnessval = self.fitnessval[ord_rerank]
                self.fitnesstst = self.fitnesstst[ord_rerank]
                self.complexity = self.complexity[ord_rerank]

                PopSorted = self.population.population.copy()
                FitnessValSorted = self.fitnessval.copy()
                FitnessTstSorted = self.fitnesstst.copy()
                ComplexitySorted = self.complexity.copy()
                
                if self.verbose == GAparsimony.DEBUG:
                    print("\nStep 2. Fitness reranked")
                    print(np.c_[self.fitnessval, self.fitnesstst, self.complexity, self.population.population.population][:10, :])
                    # input("Press [enter] to continue")


            # Keep results
            # ---------------
            self._summary[iter, :] = parsimony_summary(self) # CAMBIAR AL CAMBIAR FUNCION PARSIMONY_SUMARY
            

            # Keep Best Solution
            # ------------------
            self.bestfitnessVal = self.fitnessval[0]
            self.bestfitnessTst = self.fitnesstst[0]
            self.bestcomplexity = self.complexity[0]
            self.bestsolution = np.concatenate([[self.bestfitnessVal, self.bestfitnessTst, self.bestcomplexity],self.population[0]])
            self.bestSolList.append(self.bestsolution)
            

            # Keep elapsed time in minutes
            # ----------------------------
            tac = time.time()
            self.minutes_gen = (tac - tic) / 60.0
            self.minutes_total = self.minutes_total+self.minutes_gen
            

            # Keep this generation into the History list
            # ------------------------------------------
            if self.keep_history:
                self.history.append(pd.DataFrame(np.c_[self.population.population, self.fitnessval, self.fitnesstst, self.complexity], columns=self.population.paramsnames+self.population.colsnames+["fitnessval", "fitnesstst", "complexity"]))
            

            # Call to 'monitor' function
            # --------------------------
            if self.verbose > 0:
                parsimony_monitor(self)  
            
            if self.verbose == GAparsimony.DEBUG:
                print("\nStep 3. Fitness results")
                print(np.c_[self.fitnessval, self.fitnesstst, self.complexity, self.population.population][:10, :])
                # input("Press [enter] to continue")
            
            
            # Exit?
            # -----
            best_val_cost = self._summary[:,0][~np.isnan(self._summary[:,0])]
            if self.bestfitnessVal >= self.maxFitness:
                break
            if self.iter == self.maxiter:
                break
            if (len(best_val_cost)-(np.argmax(best_val_cost)+1)) >= self.early_stop:
                break
            
            
            # Selection Function
            # ------------------
            self._selection()

            if self.verbose == GAparsimony.DEBUG:
                print("\nStep 4. Selection")
                print(np.c_[self.fitnessval, self.fitnesstst, self.complexity, self.population.population][:10, :])
                # input("Press [enter] to continue")


            # CrossOver Function
            # ------------------
            if self.pcrossover > 0:
                nmating = int(np.floor(self.popSize/2))
                mating = np.random.choice(list(range(2 * nmating)), size=(2 * nmating), replace=False).reshape((nmating, 2))
                for i in range(nmating):
                    if self.pcrossover > np.random.uniform(low=0, high=1):
                        parents = mating[i, ]
                        self._crossover(parents=parents)
                if self.verbose == GAparsimony.DEBUG:
                    print("\nStep 5. CrossOver")
                    print(np.c_[self.fitnessval, self.fitnesstst, self.complexity, self.population.population][:10, :])
                    # input("Press [enter] to continue")


            # New generation with elitists
            # ----------------------------
            if (self.elitism > 0):
                self.population[:self.elitism] = PopSorted[:self.elitism]
                self.fitnessval[:self.elitism] = FitnessValSorted[:self.elitism]
                self.fitnesstst[:self.elitism] = FitnessTstSorted[:self.elitism]
                self.complexity[:self.elitism] = ComplexitySorted[:self.elitism]
            if (self.verbose == GAparsimony.DEBUG):
                print("\nStep 6. With Elitists")
                print(np.c_[self.fitnessval, self.fitnesstst, self.complexity, self.population.population][:10, :])
                # input("Press [enter] to continue")
            

            # Mutation function
            # -----------------
            if self.pmutation > 0:
                self._mutation() # Da problemas, es por la semilla aleatoria
                if self.verbose == GAparsimony.DEBUG:

                    print("\nStep 7. Mutation")
                    print(np.c_[self.fitnessval, self.fitnesstst, self.complexity, self.population.population][:10, :])
                    # input("Press [enter] to continue")
    
    def _rerank(self):

        cost1 = self.fitnessval.copy().astype(float)
        cost1[np.isnan(cost1)]= np.NINF

        ord = order(cost1, decreasing = True)
        cost1 = cost1[ord]
        complexity = self.complexity.copy()
        complexity[np.isnan(complexity)] = np.Inf
        complexity = complexity[ord]
        # position = range(len(cost1))
        position = ord
  
        # start
        pos1 = 0
        pos2 = 1
        cambio = False
        error_posic = self.best_score
  
        while not pos1 == self.popSize:
            # Obtaining errors
            if pos2 >= self.popSize:
                if cambio:
                    pos2 = pos1+1
                    cambio = False
                else:
                    break
            error_indiv2 = cost1[pos2]
    
            # Compare error of first individual with error_posic. Is greater than threshold go to next point
            #      if ((Error.Indiv1-error_posic) > model@rerank_error) error_posic=Error.Indiv1
        
            error_dif = abs(error_indiv2-error_posic)
            if not np.isfinite(error_dif):
                error_dif = np.Inf
            if error_dif < self.rerank_error:
        
                # If there is not difference between errors swap if Size2nd < SizeFirst
                size_indiv1 = complexity[pos1]
                size_indiv2 = complexity[pos2]
                if size_indiv2<size_indiv1:
            
                    cambio = True
                
                    swap_indiv = cost1[pos1]
                    cost1[pos1] = cost1[pos2]
                    cost1[pos2] = swap_indiv
                            
                    complexity[pos1], complexity[pos2] = complexity[pos2], complexity[pos1]

                    position[pos1], position[pos2] = position[pos2], position[pos1]
                
                    if self.verbose == GAparsimony.DEBUG:
                        print(f"SWAP!!: pos1={pos1}({size_indiv1}), pos2={pos2}({size_indiv2}), error_dif={error_dif}")
                        print("-----------------------------------------------------")
                pos2 = pos2+1

            elif cambio:
                cambio = False
                pos2 = pos1+1
            else:
                pos1 = pos1+1
                pos2 = pos1+1
                error_dif2 = abs(cost1[pos1]-error_posic)
                if not np.isfinite(error_dif2):
                    error_dif2 = np.Inf
                if error_dif2 >= self.rerank_error:
                    error_posic = cost1[pos1]

        return position
    
    def _selection(self, *args, **kwargs):
        # Establezco esta cabecera para permitir una reimplementación si se desea

        if self.selection == "linear":
            r = 2/(self.popSize*(self.popSize-1)) if "r" not in kwargs else kwargs["r"]
            q = 2/self.popSize if "q" not in kwargs else kwargs["q"]

            rank = range(self.popSize)
            prob = map(lambda x: q - (x)*r, rank)

            sel = np.random.choice(list(rank), size=self.popSize, replace=True, p=list(map(lambda x: np.min(np.ma.masked_array(np.array([max(0, x), 1]), np.isnan(np.array([max(0, x), 1])))), prob)))
        
        elif self.selection == "nlinear":
            # Nonlinear-rank selection
            # Michalewicz (1996) Genetic Algorithms + Data Structures = Evolution Programs. p. 60
            q = 0.25 if "q" not in kwargs else kwargs["q"]
            rank = list(range(self.popSize)) # population are sorted
            prob = np.array(list(map(lambda x: q*(1-q)**(x), rank)))
            prob = prob / prob.sum()
            
            sel = np.random.choice(list(rank), size=self.popSize, replace=True, p=list(map(lambda x: np.min(np.ma.masked_array(np.array([max(0, x), 1]), np.isnan(np.array([max(0, x), 1])))), prob)))
        
        elif self.selection == "random":
            sel = np.random.choice(list(range(self.popSize)), size=self.popSize, replace=True)
        
        else:
            raise Exception("Not a valid selection mode provided!!!!!!")
        
        self.population.population = self.population[sel]
        self.fitnessval = self.fitnessval[sel]
        self.fitnesstst = self.fitnesstst[sel]
        self.complexity = self.complexity[sel]
                
    def _crossover(self, parents, alpha=0.1, perc_to_swap=0.5):

        p=parents.copy()

        parents = self.population[parents]
        children = parents # Vector
        pos_param = list(range(len(self.population.paramsnames)))
        pos_features = np.array(list(range(len(self.population.paramsnames), len(self.population.paramsnames) + len(self.population.colsnames))))
        
        # Heuristic Blending for parameters
        alpha = 0.1
        Betas = np.random.uniform(size=len(self.population.paramsnames), low=0, high=1)*(1+2*alpha)-alpha  # 1+alpha*2??????
        children[0,pos_param] = parents[0,pos_param]-Betas*parents[0,pos_param]+Betas*parents[1,pos_param]  ## MAP??
        children[1,pos_param] = parents[1,pos_param]-Betas*parents[1,pos_param]+Betas*parents[0,pos_param]
        
        # Random swapping for features
        swap_param = np.random.uniform(size=len(self.population.colsnames), low=0, high=1)>=perc_to_swap
        if np.sum(swap_param)>0:
    
            features_parent1 = parents[0,pos_features]
            features_parent2 = parents[1,pos_features]
            pos_features = pos_features[swap_param]
            children[0,pos_features] = features_parent2[swap_param]
            children[1,pos_features] = features_parent1[swap_param]
  
  
        # correct params that are outside (min and max)
        thereis_min = children[0] < self.population._min
        children[0,thereis_min] = self.population._min[thereis_min]
        thereis_min = children[1] < self.population._min
        children[1,thereis_min] = self.population._min[thereis_min]
        
        thereis_max = children[0] > self.population._max
        children[0,thereis_max] = self.population._max[thereis_max]
        thereis_max = (children[1] > self.population._max)
        children[1,thereis_max] = self.population._max[thereis_max]
  
        aux = np.empty(2)
        aux[:] = np.nan

        self.population[p] = children
        self.fitnessval[p] = aux.copy()
        self.fitnesstst[p] = aux.copy()
        self.complexity[p] = aux.copy()

    def _mutation(self):

         # Uniform random mutation (except first individual)
        nparam_to_mute = round(self.pmutation*(len(self.population.paramsnames)+len(self.population.colsnames))*self.popSize)
        if nparam_to_mute<1:
            nparam_to_mute = 1
  
        for _ in range(nparam_to_mute):
  
            i = np.random.randint((self.not_muted), self.popSize, size=1)[0]
            j = np.random.randint(0, (len(self.population.paramsnames) + len(self.population.colsnames)), size=1)[0]

            value = np.random.uniform(low=self.population._min[j], high=self.population._max[j])
            self.population[i,j] = value <= self.feat_mut_thres if j>=(len(self.population.paramsnames)) else value
            
            self.fitnessval[i] = np.nan
            self.fitnesstst[i] = np.nan
            self.complexity[i] = np.nan

    def _population(self, type_ini_pop="randomLHS"):
  
        nvars = len(self.population.paramsnames) + len(self.population.colsnames)
        if type_ini_pop=="randomLHS":
            population = randomLHS(self.popSize, nvars, seed=self.seed_ini)
        elif type_ini_pop=="geneticLHS":
            population = geneticLHS(self.popSize, nvars, seed=self.seed_ini)
        elif type_ini_pop=="improvedLHS":
            population = improvedLHS(self.popSize, nvars, seed=self.seed_ini)
        elif type_ini_pop=="maximinLHS":
            population = maximinLHS(self.popSize, nvars, seed=self.seed_ini)
        elif type_ini_pop=="optimumLHS":
            population = optimumLHS(self.popSize, nvars, seed=self.seed_ini)
        elif type_ini_pop=="random":
            population = (np.random.rand(self.popSize*nvars) * (nvars - self.popSize) + self.popSize).reshape(self.popSize*nvars, 1)
  
        # Scale matrix with the parameters range
        population = population*(self.population._max-self.population._min)
        population = population+self.population._min
        # Convert features to binary 
        population[:, len(self.population.paramsnames):nvars] = population[:, len(self.population.paramsnames):nvars]<=self.feat_thres

        return population

    def __str__(self):
        print("An object of class \"ga_parsimony\"")
        print(f"Call: {self.call}")
        print("Available slots:")
        print(f"bestfitnessVal: {self.bestfitnessVal}")
        print(f"bestfitnessTst: {self.bestfitnessTst}")
        print(f"bestcomplexity: {self.bestcomplexity}")
        print(f"bestsolution: {self.bestsolution}")
        print(f"min_param: {self.population._min}")
        print(f"max_param: {self.population._max}")
        print(f"nParams: {len(self.population.paramsnames)}")
        print(f"feat_thres: {self.feat_thres}")
        print(f"feat_mut_thres: {self.feat_mut_thres}")
        print(f"not_muted: {self.not_muted}")
        print(f"rerank_error: {self.rerank_error}")
        print(f"iter_start_rerank: {self.iter_start_rerank}")
        print(f"nFeatures: {len(self.population.colsnames)}")
        print(f"names_param: {self.population.paramsnames}")
        print(f"names_features: {self.population.colsnames}")
        print(f"popSize: {self.popSize}")
        print(f"iter: {self.iter}") 
        print(f"early_stop: {self.early_stop}")
        print(f"maxiter: {self.maxiter}")
        print(f"minutes_gen: {self.minutes_gen}")
        print(f"minutes_total: {self.minutes_total}")
        print(f"suggestions: {self.suggestions}")
        print(f"population: {self.population}")
        print(f"elitism: {self.elitism}")
        print(f"pcrossover: {self.pcrossover}")
        print(f"pmutation: {self.pmutation}")
        print(f"best_score: {self.best_score}")
        print(f"solution_best_score: {self.solution_best_score}")
        print(f"fitnessval: {self.fitnessval}")
        print(f"fitnesstst: {self.fitnesstst}")
        print(f"complexity: {self.complexity}")
        print(f"summary: {self._summary}")
        print(f"bestSolList: {self.bestSolList}")

        print(f"history: ")
        for h in self.history:
            print(h)

    # def __repr__(self):
    #     print("An object of class \"ga_parsimony\"")
    #     print(f"Call: {self.call}")
    #     print("Available slots:")
    #     print(f"bestfitnessVal: {self.bestfitnessVal}")
    #     print(f"bestfitnessTst: {self.bestfitnessTst}")
    #     print(f"bestcomplexity: {self.bestcomplexity}")
    #     print(f"bestsolution: {self.bestsolution}")
    #     print(f"min_param: {self.population._min}")
    #     print(f"max_param: {self.population._max}")
    #     print(f"nParams: {len(self.population.paramsnames)}")
    #     print(f"feat_thres: {self.feat_thres}")
    #     print(f"feat_mut_thres: {self.feat_mut_thres}")
    #     print(f"not_muted: {self.not_muted}")
    #     print(f"rerank_error: {self.rerank_error}")
    #     print(f"iter_start_rerank: {self.iter_start_rerank}")
    #     print(f"nFeatures: {len(self.population.colsnames)}")
    #     print(f"names_param: {self.population.paramsnames}")
    #     print(f"names_features: {self.population.colsnames}")
    #     print(f"popSize: {self.popSize}")
    #     print(f"iter: {self.iter}") 
    #     print(f"early_stop: {self.early_stop}")
    #     print(f"maxiter: {self.maxiter}")
    #     print(f"minutes_gen: {self.minutes_gen}")
    #     print(f"minutes_total: {self.minutes_total}")
    #     print(f"suggestions: {self.suggestions}")
    #     print(f"population: {self.population}")
    #     print(f"elitism: {self.elitism}")
    #     print(f"pcrossover: {self.pcrossover}")
    #     print(f"pmutation: {self.pmutation}")
    #     print(f"best_score: {self.best_score}")
    #     print(f"solution_best_score: {self.solution_best_score}")
    #     print(f"fitnessval: {self.fitnessval}")
    #     print(f"fitnesstst: {self.fitnesstst}")
    #     print(f"complexity: {self.complexity}")
    #     print(f"summary: {self._summary}")
    #     print(f"bestSolList: {self.bestSolList}")

    #     print(f"history: ")
    #     for h in self.history:
    #         print(h)



    def summary(self, **kwargs):

        x = {"popSize" : self.popSize,
                "maxiter" : self.maxiter,
                "early_stop" : self.early_stop,
                "rerank_error" : self.rerank_error,
                "elitism" : self.elitism,
                "nParams" : len(self.population.paramsnames),
                "nFeatures" : len(self.population.colsnames),
                "pcrossover" : self.pcrossover,
                "pmutation" : self.pmutation,
                "feat_thres" : self.feat_thres,
                "feat_mut_thres" : self.feat_mut_thres,
                "not_muted" : self.not_muted,
                "domain" : np.stack([self.population._min, self.population._max], axis=0),
                "suggestions" : self.suggestions,
                "iter" : self.iter,
                "best_score" : self.best_score,
                "bestfitnessVal" : self.bestfitnessVal,
                "bestfitnessTst" : self.bestfitnessTst,
                "bestcomplexity" : self.bestcomplexity,
                "minutes_total" : self.minutes_total,
                "bestsolution" : self.bestsolution,
                "solution_best_score":self.solution_best_score}

        # Para contolar si lo está asignando
        try:
            frame = inspect.currentframe().f_back
            next_opcode = opcode.opname[frame.f_code.co_code[frame.f_lasti+2]]
            if next_opcode not in ["POP_TOP", "PRINT_EXPR"]: # Si no lo asigna
                return x 
        finally:
            del frame 
    
        head = kwargs["head"] if "head" in kwargs.keys() else 10
        tail = kwargs["tail"] if "tail" in kwargs.keys() else 1
        chead = kwargs["chead"] if "chead" in kwargs.keys() else 20
        ctail = kwargs["ctail"] if "ctail" in kwargs.keys() else 1


        print("+------------------------------------+")
        print("|             GA-PARSIMONY           |")
        print("+------------------------------------+\n")
        print("GA-PARSIMONY settings:")
        print(f" Number of Parameters      = {x['nParams']}")
        print(f" Number of Features        = {x['nFeatures']}")
        print(f" Population size           = {x['popSize']}")
        print(f" Maximum of generations    = {x['maxiter']}")
        print(f" Number of early-stop gen. = {x['early_stop']}")
        print(f" Elitism                   = {x['elitism']}")
        print(f" Crossover probability     = {x['pcrossover']}")
        print(f" Mutation probability      = {x['pmutation']}")
        print(f" Max diff(error) to ReRank = {x['rerank_error']}")
        print(f" Perc. of 1s in first popu.= {x['feat_thres']}")
        print(f" Prob. to be 1 in mutation = {x['feat_mut_thres']}")
        
        print("\n Search domain = ")
        print(pd.DataFrame(data=x["domain"], columns=self.population.paramsnames+self.population.colsnames, index=["Min_param", "Max_param"]))

        if x["suggestions"] is not None and x["suggestions"].shape[0]>0:
            print("Suggestions =")
            for m in x["suggestions"]:
                printShortMatrix(m, head, tail, chead, ctail) # Revisar


        print("\n\nGA-PARSIMONY results:")
        print(f" Iterations                = {x['iter']+1}")
        print(f" Best validation score = {x['best_score']}")
        print(f"\n\nSolution with the best validation score in the whole GA process = \n")
        print(pd.DataFrame(data=x["solution_best_score"][np.newaxis,], columns=["fitnessVal","fitnessTst","complexity"]+self.population.paramsnames+self.population.colsnames))
        
        print(f"\n\nResults of the best individual at the last generation = \n")
        print(f" Best indiv's validat.cost = {x['bestfitnessVal']}")
        print(f" Best indiv's testing cost = {x['bestfitnessTst']}")
        print(f" Best indiv's complexity   = {x['bestcomplexity']}")
        print(f" Elapsed time in minutes   = {x['minutes_total']}")
        print(f"\n\nBEST SOLUTION = \n")
        print(pd.DataFrame(data=x["bestsolution"][np.newaxis,], columns=["fitnessVal","fitnessTst","complexity"]+self.population.paramsnames+self.population.colsnames))
    

    # Plot a boxplot evolution of val cost, tst cost and complexity for the elitists
    # ------------------------------------------------------------------------------
    def plot(self, min_iter=None, max_iter=None, main_label="Boxplot cost evolution", 
                                steps=5, size_plot=None, *args):

        if (len(self.history[0])<1):
            print("'object@history' must be provided!! Set 'keep_history' to TRUE in ga_parsimony() function.")
        if not min_iter:
            min_iter = 0
        if not max_iter:
            max_iter = self.iter + 1
        
        # Preparacipon de los datos
        # ==========================
        mat_val = None
        mat_tst = None
        mat_complex = None
        for iter in range(min_iter, max_iter):
            mat_val = np.c_[mat_val, self.history[iter].fitnessval[:self.elitism]] if mat_val is not None else self.history[iter].fitnessval[:self.elitism]
            mat_tst = np.c_[mat_tst, self.history[iter].fitnesstst[:self.elitism]] if mat_tst is not None else self.history[iter].fitnesstst[:self.elitism]

            aux = np.sum(self.history[iter].values[:self.elitism,(len(self.population.paramsnames)):(len(self.population.paramsnames)+len(self.population.colsnames))], axis=1)
            mat_complex = np.c_[mat_complex, aux] if mat_complex is not None else aux

        x = list(range(min_iter, max_iter))
        mat_val = mat_val.astype(float)
        mat_tst = mat_tst.astype(float)
        mat_complex = mat_complex.astype(float)

        # Grafica
        # ======================
        fig, ax = plt.subplots() if not size_plot else plt.subplots(figsize=size_plot)

        ax = sns.boxplot(x="x", y="y", data=pd.DataFrame(dict(x=np.repeat(range(min_iter, max_iter), self.elitism), y=mat_val.T.flatten())), color=(0.275191, 0.194905, 0.496005), width=0.4)
        ax = sns.boxplot(x="x", y="y", data=pd.DataFrame(dict(x=np.repeat(range(min_iter, max_iter), self.elitism), y=mat_tst.T.flatten())), color=(0.626579, 0.854645, 0.223353), width=0.4)

        plt.suptitle(main_label, fontsize=16)
        plt.title(f"Results for the last best individual: Val={round(self.bestfitnessVal, 5)}, Test={round(self.bestfitnessTst, 5)}, Num.Features={int(mat_complex[0, -1])}")

        # Eje de la derecha
        ax = sns.lineplot(x=x, y=mat_val[0], color=(0.153364, 0.497, 0.557724), style=True, dashes=[(10, 2)])
        ax.legend([],[], frameon=False)

        ax = sns.lineplot(x=x, y=mat_tst[0], color=(0.122312, 0.633153, 0.530398), style=True, dashes=[(2, 2, 10, 2)]) # 2pt line, 2pt break, 10pt line, 2pt break
        ax.legend([],[], frameon=False)
        
        
        ax.set_ylabel("Metric")
        
        # Eje de la izquierda
        ax2 = plt.twinx()
        ax2 = sns.lineplot(x=x, y=mat_complex[0], color=(0.212395, 0.359683, 0.55171))
        

        ax2 = plt.fill_between(x = x,
                 y1 = np.min(mat_complex.T, axis=1),
                 y2 = np.max(mat_complex.T, axis=1),
                 alpha = 0.1,
                 facecolor = (0.212395, 0.359683, 0.55171))

        plt.ylabel("Number of Features of Best Indiv.")

        ax.legend(handles=[mlines.Line2D([], [], linestyle="--", color=(0.153364, 0.497, 0.557724), label='Validation METRIC of best individual'),
                            mlines.Line2D([], [], linestyle="-.", color=(0.122312, 0.633153, 0.530398), label='Testing METRIC of best individual'),
                            mlines.Line2D([], [], linestyle="-", color=(0.212395, 0.359683, 0.55171), label='Number of features of best individual'),
                            mpatches.Patch(color=(0.275191, 0.194905, 0.496005), label='Validation metric'),
                            mpatches.Patch(color=(0.626579, 0.854645, 0.223353), label='Testing metric')],
                            loc=3, # 3 o 1
                            ncol=2, borderaxespad=0.)
        
        plt.xticks(np.arange(0,max_iter, steps))
        ax.set_xlabel("Number of Generation")

        ax.set_zorder(ax2.get_zorder()+1) 
        ax.patch.set_visible(False)

        plt.show()        


    def importance(self):
    
        if len(self.history[0]) < 1:
            print("'object.history' must be provided!! Set 'keep_history' to TRUE in ga_parsimony() function.")
        min_iter = 1
        max_iter = self.iter
        
        nelitistm = self.elitism
        features_hist = None
        for iter in range(min_iter, max_iter+1):
            features_hist = np.c_[features_hist, self.history[iter][0][:nelitistm, len(self.population.paramsnames):]] ## ANALIZAR CON CUIDADO

        importance = np.mean(features_hist, axis=0)
        imp_features = 100*importance[order(importance,decreasing = True)]
        if self.verbose > 0:
            print("+--------------------------------------------+")
            print("|                  GA-PARSIMONY              |")
            print("+--------------------------------------------+\n")
            print("Percentage of appearance of each feature in elitists: \n")
            print(imp_features)

        return imp_features 