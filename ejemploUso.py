import numpy as np

from src.principal import GAparsimony

# Function to evaluate each SVM individual
# ----------------------------------------
def fitness_SVM(chromosome, *args):
    # First two values in chromosome are 'C' & 'sigma' of 'svmRadial' method
    tuneGrid = data.frame(C=chromosome[1],sigma=chromosome[2])
    
    # Next values of chromosome are the selected features (TRUE if > 0.50)
    selec_feat = chromosome[2:]>0.50
    
    # Return -Inf if there is not selected features
    if np.sum(selec_feat)<1:
        return {"kappa_val": np.NINF,"kappa_test": np.NINF,"complexity": np.Inf}
    
    # Extract features from the original DB plus response (last column)
    data_train_model = data_train.loc[: ,data_train.columns[selec_feat]]
    data_test_model = data_test.loc[: ,data_test.columns[selec_feat]]
    
    # # How to validate each individual
    # # 'repeats' could be increased to obtain a more robust validation metric. Also,
    # # 'number' of folds could be adjusted to improve the measure.
    # train_control <- trainControl(method = "repeatedcv",number = 10,repeats = 10)

    # # train the model
    # set.seed(1234)
    # model <- train(Class ~ ., data=data_train_model, trControl=train_control, 
    #                method="svmRadial", tuneGrid=tuneGrid, verbose=F)

    # # Extract kappa statistics (the repeated k-fold CV and the kappa with the test DB)
    # kappa_val <- model$results$Kappa
    # kappa_test <- postResample(pred=predict(model, data_test_model),
    #                               obs=data_test_model[,ncol(data_test_model)])[2]
    # # Obtain Complexity = Num_Features*1E6+Number of support vectors
    # complexity <- sum(selec_feat)*1E6+model$finalModel@nSV 
    
    # # Return(validation score, testing score, model_complexity)
    # vect_errors <- c(kappa_val=kappa_val,kappa_test=kappa_test,complexity=complexity)
    # return(vect_errors)
    return 0


# Ranges of size and decay
min_param = np.array([00.0001, 0.00001])
max_param = np.array([99.9999, 0.99999])
names_param = ["C","sigma"]

# ga_parsimony can be executed with a different set of 'rerank_error' values
rerank_error = 0.001



GAparsimony_model = GAparsimony(fitness=fitness_SVM,
                                  min_param=min_param,
                                  max_param=max_param,
                                  names_param=names_param,
                                  nFeatures=data_train.shape[1],
                                  names_features=data_train,
                                  keep_history = True,
                                  rerank_error = rerank_error,
                                  popSize = 40,
                                  maxiter = 100, early_stop=10,
                                  feat_thres=0.90, # Perc selected features in first generation
                                  feat_mut_thres=0.10, # Prob of a feature to be one in mutation
                                  parallel = TRUE, seed_ini = 1234)