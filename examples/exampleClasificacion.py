import pandas as pd
from sklearn.model_selection import RepeatedKFold
from sklearn.svm import SVC
from sklearn.metrics import cohen_kappa_score

from GAparsimony import GAparsimony, Population, getFitness
from GAparsimony.util import svm

df = pd.read_csv("../data/sonar_csv.csv")

rerank_error = 0.001
params = {"C":{"range": (00.0001, 99.9999), "type": Population.FLOAT}, 
            "gamma":{"range": (0.00001,0.99999), "type": Population.FLOAT}, 
            "kernel": {"value": "poly", "type": Population.CONSTANT}}

cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=123)

fitness = getFitness(SVC, cohen_kappa_score, svm, cv, regression=False, test_size=0.2, random_state=42, n_jobs=-1)


GAparsimony_model = GAparsimony(fitness=fitness,
                                  params=params,
                                  features=len(df.columns[:-1]),
                                  keep_history = True,
                                  rerank_error = rerank_error,
                                  popSize = 40,
                                  maxiter = 5, early_stop=10,
                                  feat_thres=0.90, # Perc selected features in first generation
                                  feat_mut_thres=0.10, # Prob of a feature to be one in mutation
                                  seed_ini = 1234)


GAparsimony_model.fit(df.iloc[:, :-1], df.iloc[:, -1])

GAparsimony_model.summary()

GAparsimony_model.plot()

GAparsimony_model.importance()