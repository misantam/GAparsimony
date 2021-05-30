import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import cohen_kappa_score, make_scorer

from src.gaparsimony import GAparsimony
from src.population import Population

df = pd.read_csv("C:/Users/Millan/Desktop/TFM/sonar_csv.csv")


X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df.iloc[:, -1:], test_size=0.2, random_state=42)

data_train = pd.DataFrame(X_train, columns=df.columns[:-1])
data_test = pd.DataFrame(X_test, columns=df.columns[:-1])


# ga_parsimony can be executed with a different set of 'rerank_error' values
rerank_error = 0.001

params = {"C":{"range": (00.0001, 99.9999), "type": Population.FLOAT}, 
            "gamma":{"range": (0.00001,0.99999), "type": Population.FLOAT}, 
            "kernel": {"value": "poly", "type": Population.CONSTANT}}


GAparsimony_model = GAparsimony(fitness=fitness_SVM,
                                  params=params,
                                  features=len(df.columns[:-1]),
                                  keep_history = True,
                                  rerank_error = rerank_error,
                                  popSize = 40,
                                  maxiter = 5, early_stop=10,
                                  feat_thres=0.90, # Perc selected features in first generation
                                  feat_mut_thres=0.10, # Prob of a feature to be one in mutation
                                  seed_ini = 1234)


GAparsimony_model.fit()

GAparsimony_model.summary()

aux = GAparsimony_model.summary()

# writeJSONFile("./test/outputs/replicaClasificacion.json", aux)

# print(aux)

GAparsimony_model.plot()
