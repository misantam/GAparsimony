from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from sklearn.datasets import load_boston

from GAparsimony import GAparsimony, Population, getFitness
from GAparsimony.util import linearModels

boston = load_boston()
X, y = boston.data, boston.target 
X = StandardScaler().fit_transform(X)

# ga_parsimony can be executed with a different set of 'rerank_error' values
rerank_error = 0.01

params = {"alpha":{"range": (1., 25.9), "type": Population.FLOAT}, 
            "tol":{"range": (0.0001,0.9999), "type": Population.FLOAT}}

cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=123)

fitness = getFitness(Lasso, mean_squared_error, linearModels, cv, regression=True, test_size=0.2, random_state=42, n_jobs=-1)


GAparsimony_model = GAparsimony(fitness=fitness,
                                params = params, 
                                features = boston.feature_names,
                                keep_history = True,
                                rerank_error = rerank_error,
                                popSize = 40,
                                maxiter = 5, early_stop=10,
                                feat_thres=0.90, # Perc selected features in first generation
                                feat_mut_thres=0.10, # Prob of a feature to be one in mutation
                                seed_ini = 1234)


GAparsimony_model.fit(X, y)

GAparsimony_model.summary()

aux = GAparsimony_model.summary()

GAparsimony_model.plot()