from .parsimony_monitor import parsimony_monitor, parsimony_summary
from .parsimony_functions import parsimony_population, parsimony_nlrSelection, parsimony_crossover, parsimony_mutation, parsimony_rerank
from .ga_parsimony import GaParsimony
from .parsimony_miscfun import printShortMatrix



import sys
import warnings
import numpy as np
import time

from src.ordenacion import order



# controlar inicializaciones vectores y matrices nan

def GAparsimony(fitness, min_param, max_param, nFeatures, *args,                            
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
        monitor = parsimony_monitor if bool(getattr(sys, 'ps1', sys.flags.interactive)) else False # Si se esta ejecutando en una sesión interactiva
        # print("Hola")
        # print(locals())
        # # print(locals().keys())
        # # print(locals().values())
        # print(locals()["monitor"])
        # print(callable(population))
        # print(callable(elitism))

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
        if min_param is None and max_param is None:
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
        bestSolList = np.empty(maxiter, dtype=np.object)
        # bestSolList[:] = np.nan
        # bestSolList = list() # REVISAR!!!!!!!!!!!!!!!!!
        FitnessVal_vect = np.empty(popSize)
        FitnessVal_vect[:] = np.nan
        FitnessTst_vect = np.empty(popSize)
        FitnessTst_vect[:] = np.nan
        Complexity_vect = np.empty(popSize)
        Complexity_vect[:] = np.nan

        if not object:
            # Initialize 'object'
            # -------------------
            object = GaParsimony(locals(), min_param, max_param, 
                                nParams, nFeatures, 0, 
                                population, early_stop, 0, 
                                np.NINF, fitnessSummary, bestSolList, 
                                feat_thres, feat_mut_thres, not_muted, 
                                rerank_error, iter_start_rerank,
                                names_param, names_features, popSize, 
                                maxiter, suggestions, elitism, 
                                pcrossover, None, pmutation, 
                                FitnessVal_vect, FitnessTst_vect, Complexity_vect)
            
            # First population
            # ----------------
            pop = population(object, type_ini_pop=type_ini_pop)
            
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
                                object_old.history[iter_ini][0], object_old.early_stop, minutes_total, 
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
                Results_parallel = list()
                for i in range(popSize):
                    if np.isnan(FitnessVal_vect[i]) and np.sum(pop[i, object.nParams:nvars])>0:
                        Results_parallel.append(fitness(pop[i]))
                    else:
                        Results_parallel.append(np.concatenate((FitnessVal_vect[i],FitnessTst_vect[i], Complexity_vect[i]), axis=None))

                # Results_parallel = np.array([fitness(pop[i]) if not FitnessVal_vect[i] and np.sum(pop[i,(1+object.nParams):nvars])>0 else np.r_[FitnessVal_vect[i],FitnessTst_vect[i], Complexity_vect[i]] for i in range(popSize)])
                Results_parallel = np.array(Results_parallel)
                # Extract results
                # Results_parallel = Results_parallel.reshape(((Results_parallel.shape[0]*Results_parallel.shape[1])/3, 3))
                FitnessVal_vect = Results_parallel[:, 0]
                FitnessTst_vect = Results_parallel[:, 1]
                Complexity_vect = Results_parallel[:, 2]
                
            
            np.random.seed(seed_ini*iter) if not seed_ini else np.random.seed(1234*iter)
            
            # Sort by the Fitness Value
            # ----------------------------
            # ord <- order(FitnessVal_vect, decreasing = TRUE, na.last = TRUE) 
            ord = order(FitnessVal_vect, kind='heapsort', decreasing = True, na_last = True)
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
            if np.max(FitnessVal_vect)>object.best_score:
                object.best_score = np.nanmax(FitnessVal_vect)
                object.solution_best_score = (object.best_score, 
                                                FitnessTst_vect[np.argmax(FitnessVal_vect)], 
                                                Complexity_vect[np.argmax(FitnessVal_vect)], 
                                                pop[np.argmax(FitnessVal_vect)])
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
                    input("Press [enter] to continue")


            # Keep results
            # ---------------
            fitnessSummary[iter, :] = parsimony_summary(object)
            object.summary = fitnessSummary
            
            # Keep Best Solution
            # ------------------
            object.bestfitnessVal = object.fitnessval[1]
            object.bestfitnessTst = object.fitnesstst[1]
            object.bestcomplexity = object.complexity[1]
            object.bestsolution = np.concatenate([[object.bestfitnessVal, object.bestfitnessTst, object.bestcomplexity],object.population[0]])
            # names(object@bestsolution) = c("fitnessVal","fitnessTst","complexity",object@names_param,object@names_features) # No se que hacer con esto
            object.bestSolList[iter] = object.bestsolution  
            # object.bestSolList.append(object.bestsolution ) # REVISAR!!!!!!!!!!
            
            # Keep elapsed time in minutes
            # ----------------------------
            tac = time.time()
            object.minutes_gen = (tac - tic) / 60.0
            object.minutes_total = object.minutes_total+object.minutes_gen
            
            # Keep this generation into the History list
            # ------------------------------------------
            if keep_history:
                object.history.append([object.population, object.fitnessval, object.fitnesstst, object.complexity])
            
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
            best_val_cost = object.summary[:,1][~np.isnan(object.summary[:,1])]
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
                nmating = int(np.floor(object.popSize/2))
                mating = np.random.choice(list(range(2 * nmating)), size=(2 * nmating), replace=False).reshape((nmating, 2))
                for i in range(nmating):
                    if pcrossover > np.random.uniform(low=0, high=1):
                        parents = mating[i, ]
                        Crossover = crossover(object, parents)
                        pop[parents] = Crossover["children"]
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
            if callable(mutation) and pmutation > 0:
                object = mutation(object)
                pop = object.population
                FitnessVal_vect = object.fitnessval 
                FitnessTst_vect = object.fitnesstst 
                Complexity_vect = object.complexity
            
                if verbose:

                    print("Step 7. Mutation")
                    print(np.c_[FitnessVal_vect, FitnessTst_vect, Complexity_vect, object.population][:10, :])
                    input("Press [enter] to continue")
    

        return object


#################################
#                               #
# FUTURAS FUNCIONES DE LA CLASE #
#                               #
#################################

def show(object):
    print("An object of class \"ga_parsimony\"")
    print(f"Call: {object.call}")
    print("Available slots:")

    print(f"bestfitnessVal: {object.bestfitnessVal}")
    print(f"bestfitnessTst: {object.bestfitnessTst}")
    print(f"bestcomplexity: {object.bestcomplexity}")
    print(f"bestsolution: {object.bestsolution}")
    print(f"min_param: {object.min_param}")
    print(f"max_param: {object.max_param}")
    print(f"nParams: {object.nParams}")
    print(f"feat_thres: {object.feat_thres}")
    print(f"feat_mut_thres: {object.feat_mut_thres}")
    print(f"not_muted: {object.not_muted}")
    print(f"rerank_error: {object.rerank_error}")
    print(f"iter_start_rerank: {object.iter_start_rerank}")
    print(f"nFeatures: {object.nFeatures}")
    print(f"names_param: {object.names_param}")
    print(f"names_features: {object.names_features}")
    print(f"popSize: {object.popSize}")
    print(f"iter: {object.iter}") 
    print(f"early_stop: {object.early_stop}")
    print(f"maxiter: {object.maxiter}")
    print(f"minutes_gen: {object.minutes_gen}")
    print(f"minutes_total: {object.minutes_total}")
    print(f"suggestions: {object.suggestions}")
    print(f"population: {object.population}")
    print(f"elitism: {object.elitism}")
    print(f"pcrossover: {object.pcrossover}")
    print(f"pmutation: {object.pmutation}")
    print(f"best_score: {object.best_score}")
    print(f"solution_best_score: {object.solution_best_score}")
    print(f"fitnessval: {object.fitnessval}")
    print(f"fitnesstst: {object.fitnesstst}")
    print(f"complexity: {object.complexity}")
    print(f"summary: {object.summary}")
    print(f"bestSolList: {object.bestSolList}")
    print(f"history: {object.history}")


# Clase aparte?
def summary(object):
    # varnames = object.names_param + object.names_features
    domain = np.stack([object.min_param, object.max_param], axis=0)

    out = {"popSize" : object.popSize,
            "maxiter" : object.maxiter,
            "early_stop" : object.early_stop,
            "rerank_error" : object.rerank_error,
            "elitism" : object.elitism,
            "nParams" : object.nParams,
            "nFeatures" : object.nFeatures,
            "pcrossover" : object.pcrossover,
            "pmutation" : object.pmutation,
            "feat_thres" : object.feat_thres,
            "feat_mut_thres" : object.feat_mut_thres,
            "not_muted" : object.not_muted,
            "domain" : domain,
            "suggestions" : object.suggestions,
            "iter" : object.iter,
            "best_score" : object.best_score,
            "bestfitnessVal" : object.bestfitnessVal,
            "bestfitnessTst" : object.bestfitnessTst,
            "bestcomplexity" : object.bestcomplexity,
            "minutes_total" : object.minutes_total,
            "bestsolution" : object.bestsolution,
            "solution_best_score":object.solution_best_score}
    return out


def print_summary(object, **kwargs):
    
    head = kwargs["head"] if "head" in kwargs.keys() else 10
    tail = kwargs["tail"] if "tail" in kwargs.keys() else 1
    chead = kwargs["chead"] if "chead" in kwargs.keys() else 20
    ctail = kwargs["ctail"] if "ctail" in kwargs.keys() else 1

    x = summary(object)

    

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
    print(x["domain"])

    if x["suggestions"] is not None and x["suggestions"].shape[0]>0:
        print("Suggestions =")
        for m in x["suggestions"]:
            printShortMatrix(m, head, tail, chead, ctail)
        # print(x$suggestions, digits = digits, ...)

    print("\n\nGA-PARSIMONY results:")
    print(f" Iterations                = {x['iter']+1}")
    print(f" Best validation score = {x['best_score']}")
    print(f"\n\nSolution with the best validation score in the whole GA process = \n")
    # do.call(".printShortMatrix",c(list(x$solution_best_score, digits = digits),head=length(x$solution_best_score)))
    # for m in x["solution_best_score"]:
    #     printShortMatrix(m, len(x["solution_best_score"]), tail, chead, ctail)
    # printShortMatrix(x["solution_best_score"], len(x["solution_best_score"]), tail, chead, ctail) # Modificar
    print(x["solution_best_score"])
    
    print(f"\n\nResults of the best individual at the last generation = \n")
    print(f" Best indiv's validat.cost = {x['bestfitnessVal']}")
    print(f" Best indiv's testing cost = {x['bestfitnessTst']}")
    print(f" Best indiv's complexity   = {x['bestcomplexity']}")
    print(f" Elapsed time in minutes   = {x['minutes_total']}")
    print(f"\n\nBEST SOLUTION = \n")
    # do.call(".printShortMatrix",c(list(x$bestsolution, digits = digits),head=length(x$bestsolution)))
    # for m in x["bestsolution"]:
    #     printShortMatrix(m, len(x["bestsolution"]), tail, chead, ctail)
    print(x["bestsolution"]) # Modificar igual que arriba

    #print(as.vector(x$bestsolution)) #, digits = digits, ...)
    # invisible()



# # Plot a boxplot evolution of val cost, tst cost and complexity for the elitists
# # ------------------------------------------------------------------------------
# plot.ga_parsimony <- function(x, general_cex = 0.7, min_ylim=NULL, max_ylim=NULL, 
#                               min_iter=NULL, max_iter=NULL, main_label="Boxplot cost evolution", 
#                               iter_auto_ylim=3, steps=5, pos_cost_num=-3.1,  pos_feat_num=-1.7,
#                               digits_plot=4, width_plot=12, height_plot=6, window=TRUE, ...)
# {
#   object <- x
#   if (window) dev.new(1,width = width_plot, height = height_plot)
#   if (length(object@history[[1]])<1) message("'object@history' must be provided!! Set 'keep_history' to TRUE in ga_parsimony() function.")
#   if (is.null(min_iter)) min_iter <- 1
#   if (is.null(max_iter)) max_iter <- object@iter
  
#   nelitistm <- object@elitism
#   mat_val <- NULL
#   mat_tst <- NULL
#   mat_complex <- NULL
#   for (iter in min_iter:max_iter)
#   {
#     mat_val <- cbind(mat_val, object@history[[iter]]$fitnessval[1:nelitistm])
#     mat_tst <- cbind(mat_tst, object@history[[iter]]$fitnesstst[1:nelitistm])
#     mat_complex <- cbind(mat_complex, apply(object@history[[iter]]$population[1:nelitistm,(1+object@nParams):(object@nParams+object@nFeatures)],1,sum))
                                         
#   }


#   # Plot the range of num features and the nfeatures of the best individual
#   # -----------------------------------------------------------------------
#   plot((min_iter-1):max_iter, c(NA,mat_complex[1,]), lty="dashed", type="l", lwd=1.2,xaxt="n",yaxt="n",xlab="",ylab="", bty="n", axes=FALSE, 
#        xlim=c(min_iter-1,max_iter),ylim=c(1,object@nFeatures))
#   x_pol <- c(min_iter:max_iter,max_iter:min_iter, min_iter)
#   max_pol <- apply(mat_complex,2,max)
#   min_pol <- apply(mat_complex,2,min)
#   y_pol <- c(max_pol, min_pol[length(min_pol):1],max_pol[1])
#   polygon(x_pol,y_pol,col="gray90",border="gray80")
#   lines(min_iter:max_iter, mat_complex[1,], lty="dashed")
#   mtext("Number of features of best indiv.",side=4, line=-0.5, cex=general_cex*1.65)
  
#   # Axis of side 4 (vertical right)
#   # -----------------------------------------------------------------------
#   axis_side4 <- seq(from=1,to=object@nFeatures,by=round(object@nFeatures/8));
#   if (axis_side4[length(axis_side4)]!=object@nFeatures) axis_side4 <- c(axis_side4,object@nFeatures);
#   if ((axis_side4[length(axis_side4)]-axis_side4[length(axis_side4)-1]) <= 2 && object@nFeatures>=20) axis_side4 <- axis_side4[-(length(axis_side4)-1)];
#   axis(side=4, at=axis_side4, labels=F, tick=T,lwd.ticks=0.7,tcl=-0.25, xpd=TRUE, pos=max_iter,bty="n", cex=general_cex*2)
#   mtext(axis_side4,side=4,line=pos_feat_num,at=axis_side4, cex=general_cex*1.5)
  
  
  
  
#   # Boxplot evolution
#   # ------------------
#   par(new=TRUE)
  
#   if (is.null(min_ylim)) if (!is.null(iter_auto_ylim) && iter_auto_ylim>=min_iter) min_ylim <- min(c(mat_val[,iter_auto_ylim],mat_tst[,iter_auto_ylim]),na.rm=TRUE) else min_ylim <- min(c(mat_val,mat_tst),na.rm=TRUE)
#   if (is.null(max_ylim)) max_ylim <- max(c(mat_val,mat_tst),na.rm=TRUE)
  
  
#   boxplot(mat_val,
#           col="white", xlim=c(min_iter-1,max_iter), ylim=c(min_ylim,max_ylim), 
#           xaxt = "n", xlab = "", ylab = "", border=T, axes=F,outline=F,
#           medlwd=0.75, pars=list(yaxt="n",xaxt="n", xlab = "", ylab = "", 
#                                  boxwex = 0.7, staplewex = 0.6, outwex = 0.5,lwd=0.75))
#   boxplot(mat_tst, col="lightgray", 
#           xlim=c(min_iter,(max_iter+1)),ylim=c(min_ylim,max_ylim), add=TRUE, border=T,outline=F,medlwd=0.75,
#           pars=list(yaxt="n",xaxt="n", xlab = "", ylab = "",bty="n", axes=F,
#                     boxwex = 0.7, staplewex = 0.6, outwex = 0.5,lwd=0.75))
  
#   lines(mat_val[1,],col="black",lty=1,lwd=1.8)
#   lines(mat_tst[1,],col="black",lty="dotdash",lwd=1.8)
  
#   if (window) title(main=main_label)
  
#   # Axis 
#   # -----
  
#   # Axis X
#   pos_txt_gen <- seq(from=min_iter-1,to=max_iter,by=5)
#   pos_txt_gen[1] <- 1
#   axis(side=1,at=c(min_iter:max_iter), labels=F, tick=T, lwd.ticks=0.7,  tcl= -0.25, pos=min_ylim)
#   axis(side=1,at=pos_txt_gen, labels=F, tick=T, lwd.ticks=0.7,   tcl= -0.5, pos=min_ylim)
#   mtext("Number of generation", side=1, line=1, adj=0.5, cex=general_cex*1.65)
#   mtext(paste("G.",pos_txt_gen,sep=""),side=1,line=-0.35,at=pos_txt_gen, cex=general_cex*1.5)
  
#   # Axis Y
#   as<-axis(side=2, at=round(seq(from=min_ylim,to=max_ylim,length.out=steps),3), labels=F, tick=T, 
#            lwd.ticks=0.7, tcl= -0.20, xpd=TRUE, pos=1, bty="n", cex=general_cex*2)
#   mtext("Cost", side=2, line=-2.0, adj=0.5,cex=general_cex*1.65)  
#   mtext(round(as,3), side=2, line=pos_cost_num, at=as, cex=general_cex*1.5)

#   # legend(x=pos_legend,max_ylim,c(paste0("Validation cost for best individual ('white' box plot of elitists)"),
#   #                            paste0("Testing cost of best individual ('gray' box plot of elitists)"),
#   #                            paste0("Number of features of best individual")),
#   #        lty=c("solid","dotdash","dashed"), cex=general_cex*1.4,lwd=c(1.4,1.7,1.2),
#   #        bty="n")
#   mtext(paste0("Results for the best individual:  val.cost (white)=",round(mat_val[1,max_iter],digits_plot),
#                ", tst.cost (gray)=",round(mat_tst[1,max_iter],digits_plot),
#                ", complexity=",round(mat_complex[1,max_iter],digits_plot),side=3,line=0,cex=general_cex*1.2))
#   return(list(mat_val=mat_val, mat_tst=mat_tst,  mat_complex=mat_complex))
# }

# setMethod("plot", "ga_parsimony", plot.ga_parsimony)

