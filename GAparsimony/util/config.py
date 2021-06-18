#################################################
#****************LINEAR MODELS******************#
#################################################

CLASSIF_LOGISTIC_REGRESSION = {"C":{"range": (1., 100.), "type": 1}, 
                               "tol":{"range": (0.0001,0.9999), "type": 1},
                               "solver":{"range": ('newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'), "type": 2}}
                            

CLASSIF_PERCEPTRON = {"tol":{"range": (0.0001,0.9999), "type": 1}, 
                      "alpha":{"range": (0.0001,0.9999), "type": 1},
                      "penalty":{"range": ('l2', 'l1', 'elasticnet'), "type": 2}}


REG_LASSO = {"tol":{"range": (0.0001,0.9999), "type": 1}, 
             "alpha":{"range": (1., 100.), "type": 1},
             "selection":{"range": ('cyclic', 'random'), "type": 2}}


REG_RIDGE = {"tol":{"range": (0.0001,0.9999), "type": 1}, 
             "alpha":{"range": (1., 100.), "type": 1},
             "solver":{"range": ('auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'), "type": 2}}

################################################
#*****************SVM MODELS*******************#
################################################

CLASSIF_SVC = {"C":{"range": (1.,100.), "type": 1}, 
               "alpha":{"range": (0.0001,0.9999), "type": 1},
               "kernel":{"range": ('linear', 'poly', 'rbf', 'sigmoid'), "type": 2}}


REG_SVR = {"C":{"range": (1.,100.), "type": 1}, 
           "alpha":{"range": (0.0001,0.9999), "type": 1},
           "kernel":{"range": ('linear', 'poly', 'rbf', 'sigmoid'), "type": 2}}


##################################################
#******************KNN MODELS********************#
##################################################                            

CLASSIF_KNEIGHBORSCLASSIFIER = {"n_neighbors":{"range": (2,11), "type": 0}, 
                                "p":{"range": (1, 3), "type": 0},
                                "weights":{"range": ('uniform', 'distance'), "type": 2},
                                "algorithm":{"range": ('ball_tree', 'kd_tree', "brute"), "type": 2}}


REG_KNEIGHBORSREGRESSOR = {"n_neighbors":{"range": (2,11), "type": 0}, 
                           "p":{"range": (1, 3), "type": 0},
                           "weights":{"range": ('uniform', 'distance'), "type": 2},
                           "algorithm":{"range": ('ball_tree', 'kd_tree', "brute"), "type": 2}}


##################################################
#******************MLP MODELS********************#
##################################################                            

CLASSIF_MLPCLASSIFIER = {"tol":{"range": (0.0001,0.9999), "type": 1},
                        "alpha":{"range": (0.0001, 0.999), "type": 1},
                        "activation":{"range": ('identity', 'logistic', 'tanh', 'relu'), "type": 2},
                        "solver":{"range": ('lbfgs', 'sgd', "adam"), "type": 2}}


REG_MLPREGRESSOR = {"tol":{"range": (0.0001,0.9999), "type": 1},
                    "alpha":{"range": (0.0001, 0.999), "type": 1},
                    "activation":{"range": ('identity', 'logistic', 'tanh', 'relu'), "type": 2},
                    "solver":{"range": ('lbfgs', 'sgd', "adam"), "type": 2}}


##################################################
#*************Random Forest MODELS***************#
##################################################                            

CLASSIF_RANDOMFORESTCLASSIFIER = {"n_estimators":{"range": (100,250), "type": 0},
                        "max_depth":{"range": (4, 20), "type": 0},
                        "min_samples_split":{"range": (2,25), "type": 20}}


REG_RANDOMFORESTREGRESSOR = {"n_estimators":{"range": (100,250), "type": 0},
                        "max_depth":{"range": (4, 20), "type": 0},
                        "min_samples_split":{"range": (2,25), "type": 20}}