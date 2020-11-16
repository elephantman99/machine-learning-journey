# -*- coding: utf-8 -*-
"""some helper functions."""
from linearRegression_gradientDescent import costs as cts
from leastSquares.build_polynomial import build_poly
from ridgeRegression.ridge_regression import ridge_regression
from math import ceil

import numpy as np

def kfold_iter(y, tx, k, seed=1, shuffle=True):
    """Generate a fold iterator for a dataset."""
    np.random.seed(seed)
    
    data_size = len(y)
    
    if shuffle == True:
        shuffled_indices = np.random.permutation(np.arange(data_size))
        shuffled_tx = tx[shuffled_indices]
        shuffled_y = y[shuffled_indices]
    else:
        shuffled_tx = tx
        shuffled_y = y
        
    fold_size = ceil(data_size / k)
    for fold_num in range(k):
        start_index = fold_num * fold_size
        end_index = min((fold_num+1) * fold_size, data_size)
        mask = np.array([i in np.arange(start_index, end_index) for i in range(data_size)])
        data_test = (shuffled_tx[mask], shuffled_y[mask])
        data_train = (shuffled_tx[~mask], shuffled_y[~mask])
        yield data_train, data_test

def cross_validation(y, tx, k, lambda_):
    """return the loss of ridge regression."""
    N, D = tx.shape
    
    ws = np.zeros((k, D))
    rmse_train = []
    rmse_test  = []
    for i, data in enumerate(kfold_iter(y, tx, k)):       
        data_train, data_test = data
        # get train and test data 
        tx_train, y_train = data_train
        tx_test, y_test = data_test
        # find optimal weights (ridge regression)
        loss_train, w_star = ridge_regression(y_train, tx_train, lambda_)
        loss_test = cts.ridge_mse(y_test, tx_test, w_star, lambda_)
        # update
        ws[i, :] = w_star
        rmse_train.append(loss_train)
        rmse_test.append(loss_test)
    # aggregate weights across each iteration
#     w_star = np.mean(ws, axis=0)
    
    return rmse_train, rmse_test, ws