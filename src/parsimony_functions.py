##############################################################################
#                                                                            #
#                         Parsimony GA operators                             #
#                                                                            #
##############################################################################

import numpy as np
from src.ordenacion import order

#########################################################
# parsimonyReRank: Function for reranking by complexity #
#########################################################
def parsimony_rerank(object, verbose=FALSE, *aux):

  cost1 = object.fitnessval
  cost1 = cost1.astype(float)
  cost1[np.isnan(cost1)]= - np.float32("inf")

  ord = order(cost1, decreasing = True)
  cost1 = cost1[ord]
  complexity = object.complexity
  complexity[np.isnan(cost1)] = np.float32("inf")
  complexity = complexity[ord]
  position = range(len(cost1))
  position = position[ord]
  
  # start
  pos1 = 1
  pos2 <- 2
  cambio <- FALSE
  error_posic = object.best_score
  
  while not pos1 == object.popSize:
    
    # Obtaining errors
    if pos2 > object.popSize:
      if cambio:
        pos2 = pos1+1
        cambio = False
      else:
        break
    error_indiv2 = cost1[pos2]
    
    # Compare error of first individual with error_posic. Is greater than threshold go to next point
    #      if ((Error.Indiv1-error_posic) > object@rerank_error) error_posic=Error.Indiv1
    
    error_dif = abs(error_indiv2-error_posic)
    if not np.isfinite(error_dif):
      error_dif = np.float32("inf")
    if error_dif < object.rerank_error:
      
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
        
        if verbose:
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
        error_dif2 = np.float32("inf")
      if error_dif2 >= object.rerank_error:
        error_posic = cost1[pos1]

  return position

# ---------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------



##########################################################################
# parsimony_importance: Feature Importance of elitists in the GA process #
##########################################################################
def parsimony_importance(object, verbose=False, *args):
  
  if len(object.history[[1]]) < 1:
    print("'object.history' must be provided!! Set 'keep_history' to TRUE in ga_parsimony() function.")
  min_iter = 1
  max_iter = object.iter
  
  nelitistm = object.elitism
  features_hist = None
  for iter in range(min_iter, max_iter+1):
    features_hist = np.c_(features_hist, object.history[[iter]].population[1:nelitistm, nParams:]) ## ANALIZAR CON CUIDADO

  importance = np.mean(features_hist, axis=0)
  # names(importance) <- object@names_features
  imp_features = 100*importance[order(importance,decreasing = True)]
  if verbose:
    
    # names(importance) <- object@names_features
    print("+--------------------------------------------+")
    print("|                  GA-PARSIMONY              |")
    print("+--------------------------------------------+\n")
    print("Percentage of appearance of each feature in elitists: \n")
    print(imp_features)

  return imp_features 




################################################################
# parsimony_population: Function for creating first generation #
################################################################
parsimony_population <- function(object, type_ini_pop="randomLHS", *args)

  nvars = object.nParams+object.nFeatures
  if type_ini_pop=="randomLHS":
    population = lhs::randomLHS(object.popSize,nvars)
  if type_ini_pop=="geneticLHS":
    population = lhs::geneticLHS(object.popSize,nvars)
  if type_ini_pop=="improvedLHS":
    population = lhs::improvedLHS(object.popSize,nvars) # BUSCAR LIBRERÍA
  if type_ini_pop=="maximinLHS":
    population = lhs::maximinLHS(object.popSize,nvars)
  if type_ini_pop=="optimumLHS":
    population = lhs::optimumLHS(object.popSize,nvars)
  if type_ini_pop=="random":
    population = (np.random.rand(object.popSize*nvars) * (nvars - object.popSize) + object.popSize).reshape(object.popSize*nvars, 1)
  
  # Scale matrix with the parameters range
  population = population*(object.max_param-object.min_param)
  population = population+object.min_param)
  # Convert features to binary 
  population[:, object.nParams:nvars] = population[:, object.nParams:nvars]<=object.feat_thres # No se si esto esta bien
  return population

# ---------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------





#########################################
# Function for selecting in GAparsimony #
# Note: population has been sorted      #
#       with ReRank algorithm           #
#########################################

parsimony_lrSelection <- function(object, 
                            r = 2/(object@popSize*(object@popSize-1)), 
                            q = 2/object@popSize, ...)
{
# Linear-rank selection
# Michalewicz (1996) Genetic Algorithms + Data Structures = Evolution Programs. p. 60
  rank <- 1:object@popSize # population are sorted in GAparsimony
  prob <- q - (rank-1)*r
  sel <- sample(1:object@popSize, size = object@popSize, 
                prob = pmin(pmax(0, prob), 1, na.rm = TRUE),
                replace = TRUE)
  out <- list(population = object@population[sel,,drop=FALSE],
              fitnessval = object@fitnessval[sel],
              fitnesstst = object@fitnesstst[sel],
              complexity = object@complexity[sel])
  return(out)
}

parsimony_nlrSelection <- function(object, q = 0.25, ...)
{
# Nonlinear-rank selection
# Michalewicz (1996) Genetic Algorithms + Data Structures = Evolution Programs. p. 60
  rank <- 1:object@popSize # population are sorted
  prob <- q*(1-q)^(rank-1)
  sel <- sample(1:object@popSize, size = object@popSize, 
                prob = pmin(pmax(0, prob), 1, na.rm = TRUE),
                replace = TRUE)
  out <- list(population = object@population[sel,,drop=FALSE],
              fitnessval = object@fitnessval[sel],
              fitnesstst = object@fitnesstst[sel],
              complexity = object@complexity[sel])
  return(out)
}
# ---------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------





###########################
# Functions for crossover #
###########################

parsimony_crossover <- function(object, parents, alpha=0.1, perc_to_swap=0.5, ...)
{
  parents <- object@population[parents,,drop = FALSE]
  n <- ncol(parents)
  children <- parents
  pos_param <- 1:object@nParams
  pos_features <- (1+object@nParams):(object@nParams+object@nFeatures)
  
  # Heuristic Blending for parameters
  alpha <- 0.1
  Betas <- runif(object@nParams)*(1+2*alpha)-alpha
  children[1,pos_param] <- parents[1,pos_param]-Betas*parents[1,pos_param]+Betas*parents[2,pos_param]
  children[2,pos_param] <- parents[2,pos_param]-Betas*parents[2,pos_param]+Betas*parents[1,pos_param]
  
  # Random swapping for features
  swap_param <- runif(object@nFeatures)>=perc_to_swap
  if (sum(swap_param)>0)
  {
    features_parent1 <- as.vector(parents[1,pos_features])
    features_parent2 <- as.vector(parents[2,pos_features])
    pos_features <- pos_features[swap_param]
    children[1,pos_features] <- features_parent2[swap_param]
    children[2,pos_features] <- features_parent1[swap_param]
  }
  
  # correct params that are outside (min and max)
  thereis_min <- (children[1,] < object@min_param)
  children[1,thereis_min] <- object@min_param[thereis_min]
  thereis_min <- (children[2,] < object@min_param)
  children[2,thereis_min] <- object@min_param[thereis_min]
  
  thereis_max <- (children[1,] > object@max_param)
  children[1,thereis_max] <- object@max_param[thereis_max]
  thereis_max <- (children[2,] > object@max_param)
  children[2,thereis_max] <- object@max_param[thereis_max]
  
  
  out <- list(children = children, fitnessval = rep(NA,2), 
              fitnesstst = rep(NA,2), complexity = rep(NA,2))
  return(out)
}
# ---------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------





##########################
# Functions for mutation #
##########################
parsimony_mutation <- function(object, ...)
{
  # Uniform random mutation (except first individual)
  nparam_to_mute <- round(object@pmutation*(object@nParams+object@nFeatures)*object@popSize)
  if (nparam_to_mute<1) nparam_to_mute=1
  
  for (item in seq(nparam_to_mute))
  {
    i <- sample((1+object@not_muted):object@popSize, size=1)
    j <- sample(1:(object@nParams+object@nFeatures), size = 1)
    object@population[i,j] <- runif(1, object@min_param[j], object@max_param[j])
    # If is a binary feature selection convert to binary
    if (j>=(1+object@nParams))  object@population[i,j] <- (object@population[i,j]<=object@feat_mut_thres)
    
    object@fitnessval[i] <- NA
    object@fitnesstst[i] <- NA
    object@complexity[i] <- NA
  }
  return(object)
}
