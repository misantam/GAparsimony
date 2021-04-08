from .parsimony_monitor import parsimony_monitor
from .parsimony_functions import parsimony_population, parsimony_nlrSelection, parsimony_crossover, parsimony_mutation
from .ga_parsimony import GaParsimony



import sys
import warnings
import numpy as np
import time

from src.ordenacion import order



# controlar inicializaciones vectores y matrices nan

class GAparsimony:
    def __init__(self, fitness, min_param, max_param, nFeatures, *args,                            
                          names_param=None, names_features=None,
                          object=None, iter_ini=None,
                          type_ini_pop="improvedLHS", 
                          popSize = 50, pcrossover = 0.8,  maxiter = 40, 
                          feat_thres=0.90, rerank_error = 0.0, iter_start_rerank = 0,
                          pmutation = 0.10, feat_mut_thres=0.10, not_muted=3,
                          elitism = None,
                          population = parsimony_population,
                          selection = parsimony_nlrSelection, 
                          crossover = parsimony_crossover, 
                          mutation = parsimony_mutation, 
                          keep_history = False,
                          path_name_to_save_iter = None,
                          early_stop = None, maxFitness = float("inf"), suggestions = None, 
                          parallel = False,
                        #   monitor = parsimony_monitor if sys.flags.interactive else False , # Si se esta ejecutando en una sesión interactiva
                          seed_ini = None, verbose=False):
        if not elitism:
            elitism = max(1, round(popSize * 0.20))
        if not early_stop:
            early_stop = maxiter
        monitor = bool(getattr(sys, 'ps1', sys.flags.interactive)) # Si se esta ejecutando en una sesión interactiva
        # print("Hola")
        # print(locals().keys())
        # print(locals().values())
        print(locals()["monitor"])
        print(callable(population))
        print(callable(elitism))

        # Check parameters
        # ----------------
        if not callable(population):
            # population <- get(population)
            # https://stackoverflow.com/questions/28245607/python-equivalent-of-get-in-r-use-string-to-retrieve-value-of-symbol
            population = parsimony_population
        if not callable(selection):
            # selection <- get(selection)
            selection = parsimony_nlrSelection
        if not callable(crossover):
            # crossover <- get(crossover)
            crossover = parsimony_crossover
        if not callable(mutation):
            # mutation <- get(mutation)
            mutation = parsimony_mutation
        if not fitness:
            raise Exception("A fitness function must be provided!!!")  # O usar Exception
        if not callable(fitness):
            raise Exception("A fitness function must be provided!!!")
        if popSize < 10:
            warnings.warn("The population size is less than 10!!!")
        if maxiter < 1:
            raise ValueError("The maximum number of iterations must be at least 1!!!")
        if elitism > popSize:
            raise ValueError("The elitism cannot be larger that population size.")
        if pcrossover < 0 or pcrossover > 1:
            raise ValueError("Probability of crossover must be between 0 and 1!!!")
        if pmutation < 0 or pmutation > 1:
            raise ValueError("Probability of mutation must be between 0 and 1!!!")
        if not min_param and  not max_param:
            raise ValueError("A min and max range of values must be provided!!!")
        if len(min_param)!=len(max_param):
            raise Exception("min_param and max_param must have the same length!!!")
        if not nFeatures:
            raise Exception("Number of features (nFeatures) must be provided!!!")
        if object and not object.history:
            raise Exception("'object' must be provided with 'object@history'!!!")
        
        # nvars=chromosome length
        # -----------------------
        if not type(min_param) is np.array:
            min_param = np.array(min_param)
        if not type(max_param) is np.array:
            max_param = np.array(max_param)
        nParams = len(min_param)  # min_param como np array?
        min_param = np.concatenate((min_param, np.zeros(nFeatures)), axis=0)
        max_param = np.concatenate((max_param, np.ones(nFeatures)), axis=0)
        nvars = nParams + nFeatures
        
        # Set monitor function
        # --------------------
        # Redundante
        # if type(monitor) is bool and monitor:
        #     monitor =  parsimony_monitor
        # if not monitor:
        #     monitor = False
        if monitor:
            monitor = parsimony_monitor

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
        if suggestions:
            if type(suggestions) in [list, np.array]:
                raise Exception("Provided suggestions is a vector")
            if type(suggestions) is list:
                suggestions = np.array(suggestions)
            if len(suggestions.shape) < 2 or nvars != suggestions.shape[1]:
                raise Exception("Provided suggestions (ncol) matrix do not match the number of variables (model parameters + vector with selected features) in the problem!")
            if verbose:
                print(suggestions)


        # Initial settings
        # ----------------
        i = None # Predefine esta variable para mas adelante cuidado luego se llama i.
        

        np.random.seed(seed_ini) if seed_ini else np.random.seed(1234)
        fitnessSummary = np.empty((maxiter,6*3,))
        fitnessSummary[:] = np.nan
        # colnames(fitnessSummary) <- paste0(rep(c("max","mean","q3","median","q1","min"),3),rep(c("val","tst","complex"),each=6)) # Necesario?
        bestSolList = np.empty(maxiter)
        bestSolList[:] = np.nan
        FitnessVal_vect = np.empty(popSize)
        FitnessVal_vect[:] = np.nan
        FitnessTst_vect = np.empty(popSize)
        FitnessTst_vect[:] = np.nan
        Complexity_vect = np.empty(popSize)
        Complexity_vect[:] = np.nan

        if not object:
            # Initialize 'object'
            # -------------------
            object = GaParsimony(call, min_param, max_param, 
                                nParams, nFeatures, 0, 
                                population, early_stop, minutes_total, 
                                best_score, fitnessSummary, bestSolList, 
                                feat_thres, feat_mut_thres, not_muted, 
                                rerank_error, iter_start_rerank,
                                names_param, names_features, popSize, 
                                maxiter, suggestions, elitism, 
                                pcrossover, history, pmutation, 
                                FitnessVal_vect, FitnessTst_vect, Complexity_vect)
            
            # First population
            # ----------------
            pop = Population(type_ini_pop=type_ini_pop)
            
            if object.suggestions: # Si no es null
                ng = min(object.suggestions.shape[0], popSize)
                if ng > 0:
                    pop[0:ng, :] = object.suggestions[0:ng, :]  
            object.population = pop
            
            if verbose:
                print("Step 0. Initial population")
                print(np.c_[FitnessVal_vect, FitnessTst_vect, Complexity_vect, object.population][:10, :])
                input("Press [enter] to continue")

        else:
            if verbose:
                print("There is a GAparsimony 'object'!!!")
                print(object)

            object_old = object
            iter_ini = object_old.iter if not iter_ini else min(iter_ini,object_old.iter)
            if iter_ini <= 0:
                iter_ini = 1
            print(f"Starting GA optimization with a provided GAparsimony 'object'. Using object's GA settings and its population from iter={iter_ini}.")
            
            object = GaParsimony(object_old.call, object_old.min_param, object_old.max_param, 
                                object_old.nParams, object_old.nFeatures, 0, 
                                object_old.history[[iter_ini]]["population"], object_old.early_stop, minutes_total, 
                                best_score, fitnessSummary, bestSolList, 
                                object_old.feat_thres, object_old.feat_mut_thres, object_old.not_muted, 
                                object_old.rerank_error, object_old.iter_start_rerank,
                                object_old.names_param, object_old.names_features, object_old.popSize, 
                                object_old.maxiter, object_old.suggestions, object_old.elitism, 
                                object_old.pcrossover, np.empty(object_old.maxiter), object_old.pmutation, 
                                FitnessVal_vect, FitnessTst_vect, Complexity_vect)
            
            pop = object.population

        # Main Loop
        # ---------
        # lO QUE SERÍA EL PREDICT   
        for iter in range(maxiter):
            tic = time.time()
            
            object.iter = iter
            if not parallel:

                for t in range(popSize): # seq_len equivalente a range
                    if not FitnessVal_vect[t] and np.sum(pop[t,range(1+object.nParams, nvars)])>0: # nvars +1?
                        fit = fitness(pop[t, ]) # Es una función que se la pasas en el constructor
                        FitnessVal_vect[t] = fit[1]
                        FitnessTst_vect[t] = fit[2]
                        Complexity_vect[t] = fit[3]
            else:
                # %dopar% Nos dice que se hace en paralelo
                # Results_parallel <- foreach(i = seq_len(popSize)) %dopar% 
                #     {if (is.na(FitnessVal_vect[i]) && sum(Pop[i,(1+object@nParams):nvars])>0) fitness(Pop[i, ]) else c(FitnessVal_vect[i],FitnessTst_vect[i], Complexity_vect[i])}
                Results_parallel = np.array([fitness(pop[i, :]) if not FitnessVal_vect[i] and np.sum(pop[i,(1+object.nParams):nvars])>0 else np._r(FitnessVal_vect[i],FitnessTst_vect[i], Complexity_vect[i]) for i in range(popSize)])
                # Extract results
                Results_parallel = Results_parallel.reshape(((aux.shape[0]*aux.shape[1])/3, 3))
                FitnessVal_vect = Results_parallel[:, 1]
                FitnessTst_vect = Results_parallel[:, 2]
                Complexity_vect = Results_parallel[:, 3]
                
            
            np.random.seed(seed_ini*iter) if not seed_ini else np.random.seed(1234*iter)
            
            # Sort by the Fitness Value
            # ----------------------------
            # ord <- order(FitnessVal_vect, decreasing = TRUE, na.last = TRUE) 
            ord = order(aux, kind='heapsort', decreasing = True, na_last = True)
            PopSorted = pop[ord, :]
            FitnessValSorted = FitnessVal_vect[ord]
            FitnessTstSorted = FitnessTst_vect[ord]
            ComplexitySorted = Complexity_vect[ord]
            
            object.population = PopSorted
            object.fitnessval = FitnessValSorted
            object.fitnesstst = FitnessTstSorted
            object.complexity = ComplexitySorted
            
            pop = PopSorted
            FitnessVal_vect = FitnessValSorted
            FitnessTst_vect = FitnessTstSorted
            Complexity_vect = ComplexitySorted
            if max(FitnessVal_vect)>object.best_score:
                object.best_score = np.namax(FitnessVal_vect)
                object.solution_best_score = (object.best_score, 
                                                FitnessTst_vect[np.argmax(FitnessVal_vect)], 
                                                Complexity_vect[np.argmax(FitnessVal_vect)], 
                                                Pop[np.argmax(FitnessVal_vect)])
                # names(object.solution_best_score) <- c("fitnessVal","fitnessTst","complexity",object@names_param,object@names_features)

            
            
            
            

            
            if verbose:
                print("Step 1. Fitness sorted")
                print(np.c_[FitnessVal_vect, FitnessTst_vect, Complexity_vect, object.population][:10, :])
                input("Press [enter] to continue")

            
            
            # Reorder models with ReRank function
            # -----------------------------------
            if object.rerank_error!=0.0 and object.iter>=iter_start_rerank:
                ord_rerank = parsimony_rerank(object, verbose=verbose)
                PopSorted = pop[ord_rerank]
                FitnessValSorted = FitnessVal_vect[ord_rerank]
                FitnessTstSorted = FitnessTst_vect[ord_rerank]
                ComplexitySorted = Complexity_vect[ord_rerank]
                
                object.population = PopSorted
                object.fitnessval = FitnessValSorted
                object.fitnesstst = FitnessTstSorted
                object.complexity = ComplexitySorted
                
                pop = PopSorted
                FitnessVal_vect = FitnessValSorted
                FitnessTst_vect = FitnessTstSorted
                Complexity_vect = ComplexitySorted
                
                if (verbose):
                    print("Step 2. Fitness reranked")
                    print(np.c_[FitnessVal_vect, FitnessTst_vect, Complexity_vect, object.population][:10, :])
                    readline(prompt="Press [enter] to continue")


            # Keep results
            # ---------------
            fitnessSummary[iter, :] = parsimony_summary(object)
            object.summary = fitnessSummary
            
            # Keep Best Solution
            # ------------------
            object.bestfitnessVal = object.fitnessval[1]
            object.bestfitnessTst = object.fitnesstst[1]
            object.bestcomplexity = object.complexity[1]
            object.bestsolution = np.concatenate(object.bestfitnessVal, object.bestfitnessTst, object.bestcomplexity,object.population[0, :])
            # names(object@bestsolution) = c("fitnessVal","fitnessTst","complexity",object@names_param,object@names_features) # No se que hacer con esto
            object.bestSolList[[iter]] = object.bestsolution 
            
            # Keep elapsed time in minutes
            # ----------------------------
            tac = time.time()
            object.minutes_gen = (tac - tic).total_seconds() / 60.0
            object.minutes_total = object.minutes_total+object.minutes_gen
            
            # Keep this generation into the History list
            # ------------------------------------------
            if keep_history:
                object.history[[iter]] = {"population": bject.population, "fitnessval": object.fitnessval, 
                                                            "fitnesstst": object.fitnesstst, "complexity": object.complexity}
            
            # Call to 'monitor' function
            # --------------------------
            # if path_name_to_save_iter:
            #     save(object,file=path_name_to_save_iter)
            if callable(monitor) and not verbose:
                monitor(object)  
            
            if verbose:
                print("Step 3. Fitness results")
                print(np.c_[FitnessVal_vect, FitnessTst_vect, Complexity_vect, object.population][:10, :])
                input("Press [enter] to continue")
            
            
            # Exit?
            # -----
            best_val_cost = object.summary[:,1][~numpy.isnan(object.summary[:,1])]
            if object.bestfitnessVal >= maxFitness:
                break
            if object.iter == maxiter:
                break
            if (1+len(best_val_cost)-np.argmax(best_val_cost))>=early_stop:
                break
            
            
            # Selection Function
            # ------------------
            if (callable(selection)):
                sel = selection(object)
                pop = sel["population"]
                FitnessVal_vect = sel["fitnessval"]
                FitnessTst_vect = sel["fitnesstst"]
                Complexity_vect = sel["complexity"]
            else:
                sel = np.random.choice(list(range(popSize)), size=popSize, replace=True)
                pop = object.population[sel]
                FitnessVal_vect = object.fitnessval[sel]
                FitnessTst_vect = object.fitnesstst[sel]
                Complexity_vect = object.complexity[sel]

            object.population = pop
            object.fitnessval = FitnessVal_vect
            object.fitnesstst = FitnessTst_vect
            object.complexity = Complexity_vect
            
            
            if verbose:
                print("Step 4. Selection")
                print(np.c_[FitnessVal_vect, FitnessTst_vect, Complexity_vect, object.population][:10, :])
                input("Press [enter] to continue")

            
            
            # CrossOver Function
            # ------------------
            if callable(crossover) and pcrossover > 0:
                nmating = np.floor(object.popSize/2)
                np.random.sample(list(range(2 * nmating)), size=(2 * nmating)).reshape((nmating, 2))
                for i in range(nmating):
                    if pcrossover > np.random.uniform(low=0, high=1):
                        parents = mating[i, ]
                        Crossover = crossover(object, parents)
                        Pop[parents] = Crossover["children"]
                        FitnessVal_vect[parents] = Crossover["fitnessval"]
                        FitnessTst_vect[parents] = Crossover["fitnesstst"]
                        Complexity_vect[parents] = Crossover["complexity"]
                        
                object.population = pop
                object.fitnessval = FitnessVal_vect
                object.fitnesstst = FitnessTst_vect
                object.complexity = Complexity_vect
                
                if verbose:
                    print("Step 5. CrossOver")
                    print(np.c_[FitnessVal_vect, FitnessTst_vect, Complexity_vect, object.population][:10, :])
                    input("Press [enter] to continue")

            
            # New generation with elitists
            # ----------------------------
            if (elitism > 0):
                pop[:elitism] = PopSorted[:elitism]
                FitnessVal_vect[:elitism] = FitnessValSorted[:elitism]
                FitnessTst_vect[:elitism] = FitnessTstSorted[:elitism]
                Complexity_vect[:elitism] = ComplexitySorted[:elitism]

                object.population = pop
                object.fitnessval = FitnessVal_vect
                object.fitnesstst = FitnessTst_vect
                object.complexity = Complexity_vect

                if verbose:
                    print("Step 6. With Elitists")
                    print(np.c_[FitnessVal_vect, FitnessTst_vect, Complexity_vect, object.population][:10, :])
                    input("Press [enter] to continue")
            
            
            
            # Mutation function
            # -----------------
            if callable(mutation) & pmutation > 0:
                object = mutation(object)
                pop = object.population
                FitnessVal_vect = object.fitnessval 
                FitnessTst_vect = object.fitnesstst 
                Complexity_vect = object.complexity
            
                if verbose:

                    print("Step 7. Mutation")
                    print(np.c_[FitnessVal_vect, FitnessTst_vect, Complexity_vect, object.population][:10, :])
                    input("Press [enter] to continue")
