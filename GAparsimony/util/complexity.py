# -*- coding: utf-8 -*-

"""Complexity module.

This module contains predefined complexity functions for some of the most popular models in the scikit-learn library. The models are:

* **Generic model**: Any model from any library. This complexity function will not give us the best results since it is not specific to the model.

* **linearModels**: Any of the sklearn module models `sklearn.linear_model <https://scikit-learn.org/stable/modules/classes.html?highlight=linear#module-sklearn.linear_model>`_.

* **svm**: Any of the sklearn module models `sklearn.svm <https://scikit-learn.org/stable/modules/classes.html?highlight=linear#module-sklearn.svm>`_.

* **knn**: Any of the models: `sklearn.neighbors.KNeighborsClassifier <https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier>`_ or `sklearn.neighbors.KNeighborsRegressor <https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html#sklearn.neighbors.KNeighborsRegressor>`_.

* **mlp**: Any of the models: `sklearn.neural_network.MLPClassifier <https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier>`_ or `sklearn.neural_network.MLPRegressor <https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html#sklearn.neural_network.MLPRegressor>`_.

* **randomForest**: Any of the models: `sklearn.ensemble.RandomForestRegressor <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor>`_ or `sklearn.ensemble.RandomForestClassifier <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier>`_.

* **xgboost**: The XGboost sklearn model implementation `xgboost <https://xgboost.readthedocs.io/en/latest/python/python_api.html>`_.

Complexity functions can be defined that meet the following interface.

.. highlight:: python
.. code-block:: python

    def complexity(model, nFeatures, **kwargs):
        pass
            

"""

import numpy as np

def generic(model, nFeatures, **kwargs):
    r"""
    Generic complexity function.

    Parameters
    ----------
    model : model
        The model for calculating complexity.
    nFeatures : int
        The number of features the model has been trained on.
    **kwargs : 
        A variable number of named arguments.

    Returns
    -------
    int
        The complexity of the model.

    """
    return nFeatures*1E09

def linearModels(model, nFeatures, **kwargs):
    r"""
    Complexity function for linear models.

    Parameters
    ----------
    model : model
        The model for calculating complexity.
    nFeatures : int
        The number of features the model has been trained on.
    **kwargs : 
        A variable number of named arguments.

    Returns
    -------
    int
        The complexity of the model.

    """

    int_comp = np.min((1E09-1,np.sum(model.coef_**2))) # Internal Complexity Sum of squared weigths
    return nFeatures*1E09 + int_comp

def svm(model, nFeatures, **kwargs):
    r"""
    Complexity function for SVM models.

    Parameters
    ----------
    model : model
        The model for calculating complexity.
    nFeatures : int
        The number of features the model has been trained on.
    **kwargs : 
        A variable number of named arguments.

    Returns
    -------
    int
        The complexity of the model.

    """

    int_comp = np.min((1E09-1,np.sum(model.n_support_))) # Internal Complexity
    return nFeatures*1E09 + int_comp

def knn(model, nFeatures, **kwargs):
    r"""
    Complexity function for KNN models.

    Parameters
    ----------
    model : model
        The model for calculating complexity.
    nFeatures : int
        The number of features the model has been trained on.
    **kwargs : 
        A variable number of named arguments.

    Returns
    -------
    int
        The complexity of the model.

    """

    int_comp = 1E06 * 1/model.n_neighbors   # More k less flexible
    return nFeatures*1E09 + int_comp

def mlp(model, nFeatures, **kwargs):
    r"""
    Complexity function for MLP models.

    Parameters
    ----------
    model : model
        The model for calculating complexity.
    nFeatures : int
        The number of features the model has been trained on.
    **kwargs : 
        A variable number of named arguments.

    Returns
    -------
    int
        The complexity of the model.

    """

    weights = [np.concatenate(model.intercepts_)]
    for wm in model.coefs_:
        weights.append(wm.flatten())
    weights = np.concatenate(weights) 
    int_comp = np.min((1E09-1,np.sum(weights**2)))
    return nFeatures*1E09 + int_comp

def randomForest(model, nFeatures, **kwargs):
    r"""
    Complexity function for Random Forest models.

    Parameters
    ----------
    model : model
        The model for calculating complexity.
    nFeatures : int
        The number of features the model has been trained on.
    **kwargs : 
        A variable number of named arguments.

    Returns
    -------
    int
        The complexity of the model.

    """

    num_leaves = [tree.get_n_leaves() for tree in model.estimators_]
    int_comp = np.min((1E09-1,np.mean(num_leaves))) # More leaves more complex  
    return nFeatures*1E09 + int_comp

def xgboost(model, nFeatures, **kwargs):
    r"""
    Complexity function for XGBoost model.

    Parameters
    ----------
    model : model
        The model for calculating complexity.
    nFeatures : int
        The number of features the model has been trained on.
    **kwargs : 
        A variable number of named arguments.

    Returns
    -------
    int
        The complexity of the model.

    """
    df_model = model.get_booster().trees_to_dataframe()
    df_model = df_model[df_model.Feature=='Leaf']
    mean_leaves = df_model.groupby('Tree')['Feature'].count().mean()
    num_trees = df_model.Tree.nunique()   
    int_comp = np.min((1E09-1,num_trees*mean_leaves))
    return nFeatures*1E09 + int_comp