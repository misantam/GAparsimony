from .parsimony_monitor import parsimony_monitor, parsimony_summary
from .parsimony_functions import parsimony_population, parsimony_nlrSelection, parsimony_crossover, parsimony_mutation, parsimony_rerank
from .ga_parsimony import GaParsimony
from .parsimony_miscfun import printShortMatrix

import sys
import warnings
import numpy as np
import time

from src.ordenacion import order



class GAparsimony(object):

    MONITOR = 1
    DEBUG = 2

    def __init__(self, 
                fitness, 
                min_param, 
                max_param, 
                nFeatures,                           
                names_param=None, 
                names_features=None,
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
                population = parsimony_population,
                selection = parsimony_nlrSelection, 
                crossover = parsimony_crossover, 
                mutation = parsimony_mutation, 
                keep_history = False,
                early_stop = None, 
                maxFitness = np.Inf, 
                suggestions = None, 
                parallel = False,
                seed_ini = None, 
                verbose=False,
                logger = None):

        
        self.elitism = max(1, round(popSize * 0.20)) if not elitism else elitism

        # Check parameters
        # ----------------
        if not callable(population):
            population = parsimony_population
        if not callable(selection):
            selection = parsimony_nlrSelection
        if not callable(crossover):
            crossover = parsimony_crossover
        if not callable(mutation):
            mutation = parsimony_mutation
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
        if min_param is None and max_param is None:
            raise ValueError("A min and max range of values must be provided!!!")
        if len(min_param)!=len(max_param):
            raise Exception("min_param and max_param must have the same length!!!")
        if not nFeatures:
            raise Exception("Number of features (nFeatures) must be provided!!!")
        if (suggestions is not None) or (type(suggestions) is list and len(suggestions)>0 and type(suggestions[0]) is not list) or (type(suggestions) is np.array and len(suggestions.shape) < 2):
            raise Exception("Provided suggestions is a vector")
        if (type(suggestions) is np.array) and (len(min_param) + nFeatures) != suggestions.shape[1]:
            raise Exception("Provided suggestions (ncol) matrix do not match the number of variables (model parameters + vector with selected features) in the problem!")

        self.call = locals()
        self.fitness = fitness
        self.min_param = min_param
        self.max_param = max_param
        self.nFeatures=nFeatures
        self.names_param = None if names_param else names_param
        self.names_features = names_features
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
        self.mutation = mutation
        self.crossover = crossover
        self.keep_history = keep_history
        self.early_stop = maxiter if not early_stop else early_stop
        self.maxFitness = maxFitness
        self.suggestions = np.array(suggestions) if type(suggestions) is list else suggestions
        self.parallel = parallel
        self.seed_ini = seed_ini
        self.verbose = verbose
        self.logger = None

        self.nParams = len(min_param)
        self.nvars = self.nParams + self.nFeatures

        self.iter = 0
        self.minutes_total=0
        self.best_score = np.NINF
        self.history = list()

        
        
        if not type(self.min_param) is np.array:
            self.min_param = np.array(self.min_param)
        if not type(self.max_param) is np.array:
            self.max_param = np.array(self.max_param)
            
        self.min_param = np.concatenate((self.min_param, np.zeros(self.nFeatures)), axis=0)
        self.max_param = np.concatenate((self.max_param, np.ones(self.nFeatures)), axis=0)

        if self.suggestions:
            ng = min(self.suggestions.shape[0], popSize)
            if ng > 0:
                self.population[:ng, :] = self.suggestions[:ng, :]

        self.population = population(self, type_ini_pop=type_ini_pop)
        
        
    def fit(self, iter_ini=0):

        # Initialize parallel computing
        # ----------------------
        # Start parallel computing (if needed)
        # POR IMPLEMENTAR
        # if parallel:
        #     parallel = startParallel(parallel);stopCluster <- TRUE} else {parallel <- stopCluster <- FALSE} 
        # else:
        #     stopCluster = False if type(parallel).__name__ is "cluster"  else True
        #     parallel <- startParallel(parallel)
        # on.exit(if(parallel & stopCluster) stopParallel(attr(parallel, "cluster")))
        
        
        # Get suggestions
        # ---------------
        if self.verbose == GAparsimony.DEBUG and self.suggestions:
            print(self.suggestions)


        # Initial settings
        # ----------------
        np.random.seed(self.seed_ini) if self.seed_ini else np.random.seed(1234)

        self.summary = np.empty((self.maxiter,6*3,))
        self.summary[:] = np.nan
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

            self.history = self.history[iter_ini][0]
            if self.verbose == GAparsimony.DEBUG:
                print(f"Starting GA optimization with a provided GAparsimony 'object'. Using object's GA settings and its population from iter={iter_ini}.")
        
        elif self.verbose == GAparsimony.DEBUG:
            print("\nStep 0. Initial population")
            print(np.c_[self.fitnessval, self.fitnesstst, self.complexity, self.population][:10, :])
            # input("Press [enter] to continue")

        # Main Loop
        # ---------
        # lO QUE SERÍA EL PREDICT   
        for iter in range(self.maxiter):
            tic = time.time()
            
            self.iter = iter
            if not self.parallel:

                for t in range(self.popSize):
                    if not self.fitnessval[t] and np.sum(self.population[t,range(1+self.nParams, self.nvars)])>0:
                        fit = self.fitness(self.population[t, ])
                        self.fitnessval[t] = fit[1]
                        self.fitnesstst[t] = fit[2]
                        self.complexity[t] = fit[3]
            else:
                # %dopar% Nos dice que se hace en paralelo
                # Results_parallel <- foreach(i = seq_len(popSize)) %dopar% 
                #     {if (is.na(self.fitnessval[i]) && sum(Pop[i,(1+object@nParams):nvars])>0) fitness(Pop[i, ]) else c(self.fitnessval[i],self.fitnesstst[i], self.complexity[i])}
                Results_parallel = list()
                for i in range(self.popSize):
                    if np.isnan(self.fitnessval[i]) and np.sum(self.population[i, self.nParams:self.nvars])>0:
                        Results_parallel.append(self.fitness(self.population[i]))
                    else:
                        Results_parallel.append(np.concatenate((self.fitnessval[i],self.fitnesstst[i], self.complexity[i]), axis=None))

                # Results_parallel = np.array([fitness(self.population[i]) if not self.fitnessval[i] and np.sum(self.population[i,(1+object.nParams):nvars])>0 else np.r_[self.fitnessval[i],self.fitnesstst[i], self.complexity[i]] for i in range(popSize)])
                Results_parallel = np.array(Results_parallel)
                # Extract results
                # Results_parallel = Results_parallel.reshape(((Results_parallel.shape[0]*Results_parallel.shape[1])/3, 3))
                self.fitnessval = Results_parallel[:, 0]
                self.fitnesstst = Results_parallel[:, 1]
                self.complexity = Results_parallel[:, 2]
                
            
            np.random.seed(self.seed_ini*iter) if not self.seed_ini else np.random.seed(1234*iter)
            
            # Sort by the Fitness Value
            # ----------------------------
            ord = order(self.fitnessval, kind='heapsort', decreasing = True, na_last = True)
            PopSorted = self.population[ord, :]
            FitnessValSorted = self.fitnessval[ord]
            FitnessTstSorted = self.fitnesstst[ord]
            ComplexitySorted = self.complexity[ord]
            
            self.population = PopSorted
            self.fitnessval = FitnessValSorted
            self.fitnesstst = FitnessTstSorted
            self.complexity = ComplexitySorted
            if np.max(self.fitnessval)>self.best_score:
                self.best_score = np.nanmax(self.fitnessval)
                self.solution_best_score = (self.best_score, 
                                                self.fitnesstst[np.argmax(self.fitnessval)], 
                                                self.complexity[np.argmax(self.fitnessval)], 
                                                self.population[np.argmax(self.fitnessval)])


            if self.verbose == GAparsimony.DEBUG:
                print("\nStep 1. Fitness sorted")
                print(np.c_[self.fitnessval, self.fitnesstst, self.complexity, self.population][:10, :])
                # input("Press [enter] to continue")

            
            
            # Reorder models with ReRank function
            # -----------------------------------
            if self.rerank_error != 0.0 and self.iter >= self.iter_start_rerank:
                ord_rerank = parsimony_rerank(self, verbose= self.verbose)
                PopSorted = self.population[ord_rerank]
                FitnessValSorted = self.fitnessval[ord_rerank]
                FitnessTstSorted = self.fitnesstst[ord_rerank]
                ComplexitySorted = self.complexity[ord_rerank]
                
                self.population = PopSorted
                self.fitnessval = FitnessValSorted
                self.fitnesstst = FitnessTstSorted
                self.complexity = ComplexitySorted
                
                if self.verbose == GAparsimony.DEBUG:
                    print("\nStep 2. Fitness reranked")
                    print(np.c_[self.fitnessval, self.fitnesstst, self.complexity, self.population][:10, :])
                    # input("Press [enter] to continue")


            # Keep results
            # ---------------
            self.summary[iter, :] = parsimony_summary(self) # CAMBIAR AL CAMBIAR FUNCION PARSIMONY_SUMARY
            
            # Keep Best Solution
            # ------------------
            self.bestfitnessVal = self.fitnessval[1]
            self.bestfitnessTst = self.fitnesstst[1]
            self.bestcomplexity = self.complexity[1]
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
                self.history.append([self.population, self.fitnessval, self.fitnesstst, self.complexity]) # Crear data frame
            
            # Call to 'monitor' function
            # --------------------------
            if self.verbose > 0:
                parsimony_monitor(self)  
            
            if self.verbose == GAparsimony.DEBUG:
                print("\nStep 3. Fitness results")
                print(np.c_[self.fitnessval, self.fitnesstst, self.complexity, self.population][:10, :])
                # input("Press [enter] to continue")
            
            
            # Exit?
            # -----
            best_val_cost = self.summary[:,1][~np.isnan(self.summary[:,1])]
            if self.bestfitnessVal >= self.maxFitness:
                break
            if self.iter == self.maxiter:
                break
            if (1+len(best_val_cost)-np.argmax(best_val_cost)) >= self.early_stop:
                break
            
            
            # Selection Function
            # ------------------
            if (callable(self.selection)):
                sel = self.selection(self)
                self.population = sel["population"]
                self.fitnessval = sel["fitnessval"]
                self.fitnesstst = sel["fitnesstst"]
                self.complexity = sel["complexity"]
            else:
                sel = np.random.choice(list(range(self.popSize)), size=self.popSize, replace=True)
                self.population = self.population[sel]
                self.fitnessval = self.fitnessval[sel]
                self.fitnesstst = self.fitnesstst[sel]
                self.complexity = self.complexity[sel]

            
            
            if self.verbose == GAparsimony.DEBUG:
                print("\nStep 4. Selection")
                print(np.c_[self.fitnessval, self.fitnesstst, self.complexity, self.population][:10, :])
                # input("Press [enter] to continue")

            
            
            # CrossOver Function
            # ------------------
            if callable(self.crossover) and self.pcrossover > 0:
                nmating = int(np.floor(self.popSize/2))
                mating = np.random.choice(list(range(2 * nmating)), size=(2 * nmating), replace=False).reshape((nmating, 2))
                for i in range(nmating):
                    if self.pcrossover > np.random.uniform(low=0, high=1):
                        parents = mating[i, ]
                        Crossover = self.crossover(self, parents)
                        self.population[parents] = Crossover["children"]
                        self.fitnessval[parents] = Crossover["fitnessval"]
                        self.fitnesstst[parents] = Crossover["fitnesstst"]
                        self.complexity[parents] = Crossover["complexity"]
                        
                
                if self.verbose == GAparsimony.DEBUG:
                    print("\nStep 5. CrossOver")
                    print(np.c_[self.fitnessval, self.fitnesstst, self.complexity, self.population][:10, :])
                    # input("Press [enter] to continue")

            
            # New generation with elitists
            # ----------------------------
            if (self.elitism > 0):
                self.population[:self.elitism] = PopSorted[:self.elitism]
                self.fitnessval[:self.elitism] = FitnessValSorted[:self.elitism]
                self.fitnesstst[:self.elitism] = FitnessTstSorted[:self.elitism]
                self.complexity[:self.elitism] = ComplexitySorted[:self.elitism]


                if self.verbose == GAparsimony.DEBUG:
                    print("\nStep 6. With Elitists")
                    print(np.c_[self.fitnessval, self.fitnesstst, self.complexity, self.population][:10, :])
                    # input("Press [enter] to continue")
            
            
            
            # Mutation function
            # -----------------
            if callable(self.mutation) and self.pmutation > 0:
                self = self.mutation(self)
                if self.verbose == GAparsimony.DEBUG:

                    print("\nStep 7. Mutation")
                    print(np.c_[self.fitnessval, self.fitnesstst, self.complexity, self.population][:10, :])
                    # input("Press [enter] to continue")
    


    


