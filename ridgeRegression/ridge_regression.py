# -*- coding: utf-8 -*-

from linearRegression_gradientDescent import costs as cts
from splitData.split import split_data
from leastSquares.plots import plot_train_test
from leastSquares.build_polynomial import build_poly

import numpy as np

def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    D = tx.shape[1]
    w = np.linalg.inv(tx.T @ tx + lambda_ * np.identity(D)) @ tx.T @ y
    loss = cts.ridge_mse(y, tx, w, lambda_)
    return loss, w

def ridge_regression_demo(x, y, degree, ratio, seed, shuffle):
    """ridge regression demo."""
    # define parameter
    lambdas = np.logspace(-5, 0, num=15, endpoint=True, base=10.0)
    
    # split the data into training and testing
    data_train, data_test = split_data(x, y, ratio, seed, shuffle)

    # get features and targets
    x_train, y_train = data_train
    x_train = build_poly(x_train, degree) # feature augmentation
    
    x_test, y_test = data_test
    x_test = build_poly(x_test, degree)    

    rmse_train = []
    rmse_test  = []
    for ind, lambda_ in enumerate(lambdas):
        # ridge regression with a given lambda
        loss, ws = ridge_regression(y_train, x_train, lambda_)
        
        rmse_train.append(loss)
        rmse_test.append(cts.ridge_mse(y_test, x_test, ws, lambda_))
        
        print("proportion={p}, degree={d}, lambda={l:.3f}, Training RMSE={tr:.3f}, Testing RMSE={te:.3f}".format(
               p=ratio, d=degree, l=lambda_, tr=rmse_train[ind], te=rmse_test[ind]))
        
    # Plot the obtained results
    plot_train_test(rmse_train, rmse_test, lambdas, degree)