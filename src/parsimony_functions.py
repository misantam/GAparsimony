##############################################################################
#                                                                            #
#                         Parsimony GA operators                             #
#                                                                            #
##############################################################################

import numpy as np
from src.ordenacion import order
from lhs.base import *
# from lhs.base.randomLHS import randomLHS
# from lhs.base.maximinLHS import maximinLHS
# from lhs.base.improvedLHS import improvedLHS
# from lhs.base.optimumLHS import optimumLHS
# from lhs.base.geneticLHS import geneticLHS


#########################################################
# parsimonyReRank: Function for reranking by complexity #
#########################################################
def parsimony_rerank(model, verbose=False):

  cost1 = model.fitnessval
  cost1 = cost1.astype(float)
  cost1[np.isnan(cost1)]= - np.float32("inf")

  ord = order(cost1, decreasing = True)
  cost1 = cost1[ord]
  complexity = model.complexity
  complexity[np.isnan(cost1)] = np.float32("inf")
  complexity = complexity[ord]
  position = range(len(cost1))
  # position = position[ord]
  position = ord
  
  # start
  pos1 = 0
  pos2 = 1
  cambio = False
  error_posic = model.best_score
  
  while not pos1 == model.popSize:
    
    # Obtaining errors
    if pos2 >= model.popSize:
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
      error_dif = np.float32("inf")
    if error_dif < model.rerank_error:
      
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
        
        if verbose==2:
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
      if error_dif2 >= model.rerank_error:
        error_posic = cost1[pos1]

  return position

# ---------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------



##########################################################################
# parsimony_importance: Feature Importance of elitists in the GA process #
##########################################################################
def parsimony_importance(model, verbose=False):
  
  if len(model.history[0]) < 1:
    print("'object.history' must be provided!! Set 'keep_history' to TRUE in ga_parsimony() function.")
  min_iter = 1
  max_iter = model.iter
  
  nelitistm = model.elitism
  features_hist = None
  for iter in range(min_iter, max_iter+1):
    features_hist = np.c_[features_hist, model.history[iter][0][:nelitistm, model.nParams:]] ## ANALIZAR CON CUIDADO

  importance = np.mean(features_hist, axis=0)
  # names(importance) <- model@names_features
  imp_features = 100*importance[order(importance,decreasing = True)]
  if verbose:
    
    # names(importance) <- model@names_features
    print("+--------------------------------------------+")
    print("|                  GA-PARSIMONY              |")
    print("+--------------------------------------------+\n")
    print("Percentage of appearance of each feature in elitists: \n")
    print(imp_features)

  return imp_features 




################################################################
# parsimony_population: Function for creating first generation #
################################################################
def parsimony_population(model, type_ini_pop="randomLHS"):
  
  nvars = model.nParams+model.nFeatures
  if type_ini_pop=="randomLHS":
    population = randomLHS(model.popSize,nvars, seed=model.seed_ini)
  elif type_ini_pop=="geneticLHS":
    population = geneticLHS(model.popSize,nvars, seed=model.seed_ini)
  elif type_ini_pop=="improvedLHS":
    population = improvedLHS(model.popSize,nvars, seed=model.seed_ini) # BUSCAR LIBRERÍA
  elif type_ini_pop=="maximinLHS":
    population = maximinLHS(model.popSize,nvars, seed=model.seed_ini)
  elif type_ini_pop=="optimumLHS":
    population = optimumLHS(model.popSize,nvars, seed=model.seed_ini)
  elif type_ini_pop=="random":
    if model.seed_ini:
        np.random.seed(model.seed_ini)
    population = (np.random.rand(model.popSize*nvars) * (nvars - model.popSize) + model.popSize).reshape(model.popSize*nvars, 1)
  
  # Scale matrix with the parameters range
  population = population*(model.max_param-model.min_param)
  population = population+model.min_param
  # Convert features to binary 
  population[:, model.nParams:nvars] = population[:, model.nParams:nvars]<=model.feat_thres # No se si esto esta bien
  return population

# ---------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------





#########################################
# Function for selecting in GAparsimony #
# Note: population has been sorted      #
#       with ReRank algorithm           #
#########################################

def parsimony_lrSelection(model, r = None, q = None):
  
  if model.seed_ini:
    np.random.seed(model.seed_ini)

  if not r:
    r = 2/(model.popSize*(model.popSize-1))
  if not q:
    q = 2/model.popSize

  rank = range(model.popSize)
  prob = map(lambda x: q - (x)*r, rank)

  sel = np.random.choice(list(rank), size=model.popSize, replace=True, p=list(map(lambda x: np.min(np.ma.masked_array(np.array([max(0, x), 1]), np.isnan(np.array([max(0, x), 1])))), prob)))
  
  out = {"population" : model.population[sel],
          "fitnessval" : model.fitnessval[sel],
          "fitnesstst" : model.fitnesstst[sel],
          "complexity" : model.complexity[sel]}
  return out


def parsimony_nlrSelection(model, q = 0.25):
# Nonlinear-rank selection
# Michalewicz (1996) Genetic Algorithms + Data Structures = Evolution Programs. p. 60
  rank = list(range(model.popSize)) # population are sorted
  prob = np.array(list(map(lambda x: q*(1-q)**(x), rank)))
  prob = prob / prob.sum()
  
  sel = np.random.choice(list(rank), size=model.popSize, replace=True, p=list(map(lambda x: np.min(np.ma.masked_array(np.array([max(0, x), 1]), np.isnan(np.array([max(0, x), 1])))), prob)))

  out = {"population" : model.population[sel],
              "fitnessval" : model.fitnessval[sel],
              "fitnesstst" : model.fitnesstst[sel],
              "complexity" : model.complexity[sel]}
  return out

# ---------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------





###########################
# Functions for crossover #
###########################

#En esta tengo dudas de si lo hace bien
def parsimony_crossover(model, parents, alpha=0.1, perc_to_swap=0.5):

  parents = model.population[parents]
  children = parents # Vector
  pos_param = list(range(model.nParams))
  pos_features = np.array(list(range(model.nParams, model.nParams+model.nFeatures)))
  
  # Heuristic Blending for parameters
  alpha = 0.1
  Betas = np.random.uniform(size=model.nParams, low=0, high=1)*(2*alpha)-alpha  # 1+alpha*2??????
  children[0,pos_param] = parents[0,pos_param]-Betas*parents[0,pos_param]+Betas*parents[1,pos_param]  ## MAP??
  children[1,pos_param] = parents[1,pos_param]-Betas*parents[1,pos_param]+Betas*parents[0,pos_param]
  
  # Random swapping for features
  swap_param = np.random.uniform(size=model.nFeatures, low=0, high=1)>=perc_to_swap
  if np.sum(swap_param)>0:
    
    features_parent1 = parents[0,pos_features]
    features_parent2 = parents[1,pos_features]
    pos_features = pos_features[swap_param]
    children[0,pos_features] = features_parent2[swap_param]
    children[1,pos_features] = features_parent1[swap_param]
  
  
  # correct params that are outside (min and max)
  thereis_min = children[0] < model.min_param
  children[0,thereis_min] = model.min_param[thereis_min]
  thereis_min = children[1] < model.min_param
  children[1,thereis_min] = model.min_param[thereis_min]
  
  thereis_max = children[0] > model.max_param
  children[0,thereis_max] = model.max_param[thereis_max]
  thereis_max = (children[1] > model.max_param)
  children[1,thereis_max] = model.max_param[thereis_max]
  

  aux = np.empty(2)
  aux[:] = np.nan
  out = {"children" : children, "fitnessval" : aux.copy(), 
              "fitnesstst" : aux.copy(), "complexity" : aux.copy()}
  return out

# ---------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------





##########################
# Functions for mutation #
##########################
def parsimony_mutation(model):

  if model.seed_ini:
    np.random.seed(model.seed_ini)

  # Uniform random mutation (except first individual)
  nparam_to_mute = round(model.pmutation*(model.nParams+model.nFeatures)*model.popSize)
  if nparam_to_mute<1:
    nparam_to_mute = 1
  
  for _ in range(nparam_to_mute):
  
    i = np.random.randint((model.not_muted), model.popSize, size=1)[0]
    j = np.random.randint(0, (model.nParams+model.nFeatures), size=1)[0]
    model.population[i,j] = np.random.uniform(low=model.min_param[j], high=model.max_param[j])
    # If is a binary feature selection convert to binary
    if j>=(model.nParams):
      model.population[i,j] = model.population[i,j] <= model.feat_mut_thres
    
    model.fitnessval[i] = np.nan
    model.fitnesstst[i] = np.nan
    model.complexity[i] = np.nan

  return model 

