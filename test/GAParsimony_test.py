import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
import xgboost as xgb
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.preprocessing import StandardScaler

from sklearn.datasets import load_boston

from src.gaparsimony import GAparsimony

import pytest



def comparaDiccionarios(a, b):
    for x in a:
        if x is not "minutes_total" and a[x] != b[x]:
            return False
    return True

@pytest.mark.parametrize("resultado", 
{
    "popSize": 64, 
    "maxiter": 10, 
    "early_stop": 3, 
    "rerank_error": 0.01, 
    "elitism": 16, 
    "nParams": 7, 
    "nFeatures": 13, 
    "pcrossover": 0.8, 
    "pmutation": 0.1, 
    "feat_thres": 0.9, 
    "feat_mut_thres": 0.1, 
    "not_muted": 3, 
    "domain": [[10.0, 2.0, 1.0, 0.0, 0.0, 0.6, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
                [2000.0, 20.0, 20.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]], 
    "suggestions": None, 
    "iter": 9, 
    "best_score": -90.82037822898727, 
    "bestfitnessVal": -90.82037822898727, 
    "bestfitnessTst": -79.00283163570364, 
    "bestcomplexity": 10000100.0, 
    "minutes_total": 58.0256983200709, 
    "bestsolution": np.array([-90.82037822898727, -79.00283163570364, 10000100.0, 967.0049860264921, 9.045030471862425, 1.5936203632023573, 0.027887215893591777, 0.009832486468385731, 0.646748968801112, 0.9000711822947935, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0]), 
    "solution_best_score": np.array([-90.82037822898727, -79.00283163570364, 10000100.0, 967.0049860264921, 9.045030471862425, 1.5936203632023573, 0.027887215893591777, 0.009832486468385731, 0.646748968801112, 0.9000711822947935, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0])}
)
def test_GAParsimony_regresion_boston(resultado):
    boston = load_boston()
    X, y = boston.data, boston.target 
    X = StandardScaler().fit_transform(X)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    data_train = pd.DataFrame(X_train, columns=boston.feature_names)
    data_test = pd.DataFrame(X_test, columns=boston.feature_names)


    def fitness_XGBoost(chromosome):
        # First two values in chromosome are 'C' & 'sigma' of 'svmRadial' method
        tuneGrid = {
                    "n_estimators ": int(chromosome[0]),
                    "max_depth": int(chromosome[1]),
                    "min_child_weight": int(chromosome[2]),
                    "reg_alpha": chromosome[3],
                    "reg_lambda": chromosome[4],
                    "subsample": chromosome[5],
                    "colsample_bytree": chromosome[6],
                    "learning_rate": 0.01,
                    "random_state": 1234,
                    "verbosity": 0}
        
        # Next values of chromosome are the selected features (TRUE if > 0.50)
        selec_feat = chromosome[7:]>0.50
        
        # Return -Inf if there is not selected features
        if np.sum(selec_feat)<1:
            return np.array([np.NINF, np.NINF, np.Inf])
        
        # Extract features from the original DB plus response (last column)
        data_train_model = data_train.loc[: , data_train.columns[selec_feat]] 
        data_test_model = data_test.loc[: , data_test.columns[selec_feat]] 
        
        # How to validate each individual
        # 'repeats' could be increased to obtain a more robust validation metric. Also,
        # 'number' of folds could be adjusted to improve the measure.
        train_control = RepeatedKFold(n_splits=10, n_repeats=5, random_state=123)

        # train the model
    #     np.random.seed(1234)

        aux = xgb.XGBRegressor(**tuneGrid)

        model = cross_val_score(aux, data_train_model, y_train, scoring="neg_mean_squared_error", cv=train_control, n_jobs=-1)

        

        # Extract kappa statistics (the repeated k-fold CV and the kappa with the test DB)
        rmse_val = model.mean()

        model = xgb.XGBRegressor(**tuneGrid).fit(data_train_model, y_train)

        rmse_test = mean_squared_error(model.predict(data_test_model), y_test)
        # Obtain Complexity = Num_Features*1E6+Number of support vectors
        complexity = np.sum(selec_feat)*1E6 + len(model.get_booster().get_dump())
        
        # Return(validation score, testing score, model_complexity)
        return np.array([rmse_val, -rmse_test, complexity])


    # Ranges of size and decay
    min_param = np.array([10, 2, 1, 0., 0., 0.6, 0.8])
    max_param = np.array([2000, 20, 20, 1., 1., 1., 1.])
    names_param = ["n_estimators(nrounds)","max_depth", "min_child_weight", 
                "reg_alpha(lasso)", "reg_lambda(ridge)", "subsample",
                "colsample_bytree"]

    # ga_parsimony can be executed with a different set of 'rerank_error' values
    rerank_error = 0.01


    GAparsimony_model = GAparsimony(fitness=fitness_XGBoost,
                                    min_param=min_param,
                                    max_param=max_param,
                                    names_param=names_param,
                                    nFeatures=len(boston.feature_names),
                                    names_features=boston.feature_names,
                                    keep_history = True,
                                    rerank_error = rerank_error,
                                    popSize = 64,
                                    elitism = 16,
                                    maxiter = 10, early_stop=3,
                                    feat_thres=0.90, # Perc selected features in first generation
                                    feat_mut_thres=0.10, # Prob of a feature to be one in mutation
                                    parallel = True, seed_ini = 1234,
                                    verbose=GAparsimony.MONITOR)


    GAparsimony_model.fit()

    aux = GAparsimony_model.summary()



    assert comparaDiccionarios(resultado, aux)

# @pytest.mark.parametrize("shape", [
#     (2, 2),
#     (6, 6),
#     (3, 8)
# ])
# def test_randomLHS(shape):
#     assert isValidLHS(randomLHS(*shape))

