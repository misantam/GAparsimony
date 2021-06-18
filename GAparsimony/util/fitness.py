# -*- coding: utf-8 -*-

from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold
from sklearn.metrics import make_scorer
import numpy as np

def getFitness(model, metric, complexity, cv=None, regression=True, test_size=0.2, random_state=42, n_jobs=-1):
    r"""
    Generating function of a fitness function.

    Parameters
    ----------
    model : object
        The model we want to optimize. 
    metric : function
        A function that computes the metric.
    complexity : function
        A function that calculates the complexity of the model. You can use for some models those defined in `GAparsimony.util.complexity`
    cv : object, optional
        An `sklearn.model_selection`, Splitter Classes. If `None`, `sklearn.model_selection.RepeatedKFold`. Default `None`
    regression : bool, optional
        If it is a regression model `True` else `False`.
    test_size : float, optional
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split. If int, represents the
        absolute number of test samples. If None, the value is set to the
        complement of the train size. Default 0.2.
    random_state : int, optional
        Controls the shuffling applied to the data before applying the split.
        Pass an int for reproducible output across multiple function calls.
        Default `42`
    n_jobs : int, optional
        Number of jobs to run in parallel. Training the estimator and computing
        the score are parallelized over the cross-validation splits.
        ``-1`` means using all processors. Default `-1`
    Examples
    --------
    Usage example for a regression model 
    
    .. highlight:: python
    .. code-block:: python

        from sklearn.svm import SVC
        from sklearn.metrics import cohen_kappa_score

        from GAparsimony import getFitness
        from GAparsimony.util import svm

        fitness = getFitness(SVC, cohen_kappa_score, svm, cv, regression=False, test_size=0.2, random_state=42, n_jobs=-1)
    """

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

            if regression:
                fitness_val = -fitness_val
                fitness_test = -fitness_test

            return np.array([fitness_val, fitness_test, complexity(modelo, np.sum(cromosoma.columns))]), modelo
        except Exception as e:    
            print(e)
            return np.array([np.NINF, np.NINF, np.Inf]), None

    return fitness