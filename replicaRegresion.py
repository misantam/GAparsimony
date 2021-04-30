import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.preprocessing import StandardScaler
from functools import reduce

from sklearn.datasets import load_boston

from src.gaparsimony import GAparsimony




boston = load_boston()
X, y = boston.data, boston.target 
X = StandardScaler().fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

data_train = pd.DataFrame(X_train, columns=boston.feature_names)
data_test = pd.DataFrame(X_test, columns=boston.feature_names)


def fitness_NNET(chromosome):
    # First two values in chromosome are 'C' & 'sigma' of 'svmRadial' method
    tuneGrid = {"alpha": chromosome[0],"tol": chromosome[1]}
    
    # Next values of chromosome are the selected features (TRUE if > 0.50)
    selec_feat = chromosome[2:]>0.50
    
    # Return -Inf if there is not selected features
    if np.sum(selec_feat)<1:
        return np.array([np.NINF, np.NINF, np.Inf])
    
    # Extract features from the original DB plus response (last column)
    data_train_model = data_train.loc[: , boston.feature_names[selec_feat]] 
    data_test_model = data_test.loc[: , boston.feature_names[selec_feat]] 
    
    # How to validate each individual
    # 'repeats' could be increased to obtain a more robust validation metric. Also,
    # 'number' of folds could be adjusted to improve the measure.
    train_control = RepeatedKFold(n_splits=10, n_repeats=5, random_state=123)

    # train the model
    np.random.seed(1234)

    aux = Lasso(**tuneGrid)

    model = cross_val_score(aux, data_train_model, y_train, scoring="neg_mean_squared_error", cv=train_control, n_jobs=-1)

    

    # Extract kappa statistics (the repeated k-fold CV and the kappa with the test DB)
    rmse_val = model.mean()

    model = Lasso(**tuneGrid).fit(data_train_model, y_train)

    rmse_test = mean_squared_error(model.predict(data_test_model), y_test)
    # Obtain Complexity = Num_Features*1E6+Number of support vectors
    coef = 0
    for c in model.coef_:
        coef += np.sum(np.power(c, 2))
    complexity = np.sum(selec_feat)*1E6 + coef
    
    # Return(validation score, testing score, model_complexity)
    return np.array([rmse_val, -rmse_test, complexity])


# Ranges of size and decay
min_param = np.array([1., 0.0001])
max_param = np.array([25, 0.9999])
names_param = ["alpha","tol"]

# ga_parsimony can be executed with a different set of 'rerank_error' values
rerank_error = 0.01


GAparsimony_model = GAparsimony(fitness=fitness_NNET,
                                  min_param=min_param,
                                  max_param=max_param,
                                  names_param=names_param,
                                  nFeatures=len(boston.feature_names),
                                  names_features=boston.feature_names,
                                  keep_history = True,
                                  rerank_error = rerank_error,
                                  popSize = 40,
                                  maxiter = 2, early_stop=10,
                                  feat_thres=0.90, # Perc selected features in first generation
                                  feat_mut_thres=0.10, # Prob of a feature to be one in mutation
                                  parallel = True, seed_ini = 1234,
                                  verbose=GAparsimony.MONITOR)


GAparsimony_model.fit()

GAparsimony_model.summary()

aux = GAparsimony_model.summary()

# print(aux)

GAparsimony_model.plot()