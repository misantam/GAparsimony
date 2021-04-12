import numpy as np

class GaParsimony:

    def __init__(self, call, min_param, max_param, nParams, nFeatures, iter, population, early_stop, minutes_total, best_score, summary, bestSolList, feat_thres=0.90,
                  feat_mut_thres=0.1, not_muted=3, 
                  rerank_error=0., iter_start_rerank=0,
                  names_param = None,
                  names_features = None, 
                  popSize = 50, maxiter = 40, 
                  suggestions = None, elitism = None, 
                  pcrossover = 0.8,
                  history = None,
                  pmutation = 0.1, 
                  fitnessval = None, fitnesstst=None, complexity=None):

        self.call = call
        self.min_param = min_param
        self.max_param = max_param
        self.nParams = nParams
        self.feat_thres=feat_thres
        self.feat_mut_thres=feat_mut_thres
        self.not_muted=not_muted
        self.rerank_error=rerank_error
        self.iter_start_rerank=iter_start_rerank
        self.nFeatures=nFeatures
        self.names_param = None if names_param else names_param
        self.names_features = names_features
        self.popSize = popSize
        self.iter = 0
        self.early_stop = maxiter if not early_stop else early_stop
        self.maxiter = maxiter
        self.suggestions = suggestions
        self.population = None
        self.elitism = max(1, round(popSize * 0.20)) if not elitism else elitism
        self.pcrossover = pcrossover
        self.minutes_total=0
        self.best_score = -np.float32("inf")
        self.history = list() if not history else history
        self.pmutation = pmutation
        self.fitnessval = fitnessval
        self.fitnesstst=fitnesstst
        self.complexity=complexity
        self.summary = summary
        self.bestSolList = bestSolList