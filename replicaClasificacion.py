import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import cohen_kappa_score, make_scorer

from src.gaparsimony import GAparsimony
from src.population import Population

from test.utilTest import writeJSONFile

df = pd.read_csv("C:/Users/Millan/Desktop/TFM/sonar_csv.csv")


X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df.iloc[:, -1:], test_size=0.2, random_state=42)

data_train = pd.DataFrame(X_train, columns=df.columns[:-1])
data_test = pd.DataFrame(X_test, columns=df.columns[:-1])

def fitness_SVM(chromosome):
    
    # Next values of chromosome are the selected features (TRUE if > 0.50)
    selec_feat = chromosome.columns>0.50
    
    # Return -Inf if there is not selected features
    if np.sum(selec_feat)<1:
        return np.array([np.NINF, np.NINF, np.Inf])
    
    # Extract features from the original DB plus response (last column)
    data_train_model = data_train.loc[: , data_train.columns[selec_feat]] 
    data_test_model = data_test.loc[: , data_test.columns[selec_feat]] 
    
    # How to validate each individual
    # 'repeats' could be increased to obtain a more robust validation metric. Also,
    # 'number' of folds could be adjusted to improve the measure.
    train_control = RepeatedKFold(n_splits=10, n_repeats=10, random_state=123)

    # train the model
    np.random.seed(1234)

    aux = SVC(**chromosome.params)

    model = cross_val_score(aux, data_train_model, y_train.values.ravel(), scoring=make_scorer(cohen_kappa_score), cv=train_control, n_jobs=-1)

    

    # Extract kappa statistics (the repeated k-fold CV and the kappa with the test DB)
    kappa_val = model.mean()

    model = SVC(**chromosome.params).fit(data_train_model, y_train.values.ravel())

    kappa_test = cohen_kappa_score(model.predict(data_test_model), y_test)
    # Obtain Complexity = Num_Features*1E6+Number of support vectors
    complexity = np.sum(selec_feat)*1E6 + model.support_vectors_.shape[0]
    
    # Return(validation score, testing score, model_complexity)
    return np.array([kappa_val, kappa_test, complexity])


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
