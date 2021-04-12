import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import cohen_kappa_score, make_scorer

from src.principal import GAparsimony, print_summary

df = pd.read_csv("C:/Users/Millan/Desktop/TFM/sonar_csv.csv")
print(df.shape)

print(df.iloc[:, :-1].shape)


X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df.iloc[:, -1:], test_size=0.2, random_state=42)

data_train = pd.DataFrame(np.concatenate([X_train, y_train], axis=1), columns=df.columns)
data_test = pd.DataFrame(np.concatenate([X_test, y_test], axis=1), columns=df.columns)

def fitness_SVM(chromosome):
    # First two values in chromosome are 'C' & 'sigma' of 'svmRadial' method
    tuneGrid = {"C": chromosome[0],"gamma": chromosome[1]}
    
    # Next values of chromosome are the selected features (TRUE if > 0.50)
    selec_feat = chromosome[2:]>0.50
    
    # Return -Inf if there is not selected features
    if np.sum(selec_feat)<1:
        return np.array([np.NINF, np.NINF, np.Inf])
    
    # Extract features from the original DB plus response (last column)
    data_train_model = data_train.loc[: , df.columns[np.concatenate([selec_feat, True], axis=None)]] # En el original añaden true para añadir la columna target
    data_test_model = data_test.loc[: , df.columns[np.concatenate([selec_feat, True], axis=None)]] # En el original añaden true para añadir la columna target
    
    # How to validate each individual
    # 'repeats' could be increased to obtain a more robust validation metric. Also,
    # 'number' of folds could be adjusted to improve the measure.
    train_control = RepeatedKFold(n_splits=10, n_repeats=10, random_state=123)

    # train the model
    np.random.seed(1234)

    aux = SVC(**tuneGrid)

    model = cross_val_score(aux, data_train_model.iloc[:, :-1].values, data_train_model.iloc[:, -1:].values, scoring=make_scorer(cohen_kappa_score), cv=train_control, n_jobs=-1)

    

    # Extract kappa statistics (the repeated k-fold CV and the kappa with the test DB)
    kappa_val = model.mean()

    model = SVC(**tuneGrid).fit(data_train_model.iloc[:, :-1], data_train_model.iloc[:, -1:])

    kappa_test = cohen_kappa_score(model.predict(data_test_model.iloc[:, :-1]), data_test_model.iloc[:, -1:])
    # Obtain Complexity = Num_Features*1E6+Number of support vectors
    complexity = np.sum(selec_feat)*1E6 + model.support_vectors_.shape[0]
    
    # Return(validation score, testing score, model_complexity)
    return np.array([kappa_val, kappa_test, complexity])


# Ranges of size and decay
min_param = np.array([00.0001, 0.00001])
max_param = np.array([99.9999, 0.99999])
names_param = ["C","gamma"]

# ga_parsimony can be executed with a different set of 'rerank_error' values
rerank_error = 0.001


GAparsimony_model = GAparsimony(fitness=fitness_SVM,
                                  min_param=min_param,
                                  max_param=max_param,
                                  names_param=names_param,
                                  nFeatures=len(df.columns[:-1]),
                                  names_features=df.columns[:-1],
                                  keep_history = True,
                                  rerank_error = rerank_error,
                                  popSize = 40,
                                  maxiter = 1, early_stop=10,
                                  feat_thres=0.90, # Perc selected features in first generation
                                  feat_mut_thres=0.10, # Prob of a feature to be one in mutation
                                  parallel = True, seed_ini = 1234)


print(GAparsimony_model)

print_summary(GAparsimony_model)
