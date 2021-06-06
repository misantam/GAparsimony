# -*- coding: utf-8 -*-

from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold
from sklearn.metrics import make_scorer
import numpy as np

def getFitness(model, metric, complexity, cv, regresion=True, test_size=0.2, random_state=42, n_jobs=-1): #X, y, 
    
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    if model is None:
        raise Exception("A model class must be provided!!!")
    if metric is None or not callable(metric):
        raise Exception("A metric function must be provided!!!")
    if complexity is None or not callable(complexity):
        raise Exception("A complexity function must be provided!!!")
    if cv is None:
        cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=random_state)
    
    def fitness(cromosoma, **kwargs):
        if "pandas" in str(type(kwargs["X"])):
            kwargs["X"] = kwargs["X"].values
        if "pandas" in str(type(kwargs["y"])):
            kwargs["y"] = kwargs["y"].values
        X_train, X_test, y_train, y_test = train_test_split(kwargs["X"], kwargs["y"], test_size=test_size, random_state=random_state)
        try:
            # Extract features from the original DB plus response (last column)
            data_train_model = X_train[: , cromosoma.columns] 
            data_test_model = X_test[: , cromosoma.columns] 

            # train the model

            aux = model(**cromosoma.params)
            fitness_val = cross_val_score(aux, data_train_model, y_train, scoring=make_scorer(metric), cv=cv, n_jobs=n_jobs).mean()
            modelo = model(**cromosoma.params).fit(data_train_model, y_train)
            fitness_test = metric(modelo.predict(data_test_model), y_test)

            if regresion:
                fitness_val = -fitness_val
                fitness_test = -fitness_test

            return np.array([fitness_val, fitness_test, complexity(modelo, cromosoma.columns)]), modelo
        except Exception as e:    
            print(e)
            return np.array([np.NINF, np.NINF, np.Inf]), None

    return fitness