# -*- coding: utf-8 -*-
"""some helper functions."""

import numpy as np
from math import ceil

def split_data(x, y, ratio, seed=1, shuffle=True):
    """
    split the dataset based on the split ratio. If ratio is 0.8 
    you will have 80% of your data set dedicated to training 
    and the rest dedicated to testing
    """
    # set seed
    np.random.seed(seed)
    
    data_size = len(y)
    
    if shuffle:
        shuffled_indices = np.random.permutation(np.arange(data_size))
        shuffled_x = x[shuffled_indices]
        shuffled_y = y[shuffled_indices]
    else:
        shuffled_x = x
        shuffled_y = y
    
    cut = ceil(data_size * ratio)
    x_train, y_train = shuffled_x[:cut], shuffled_y[:cut]
    x_test, y_test   = shuffled_x[cut:], shuffled_y[cut:]
    
    return (x_train, y_train), (x_test, y_test)

def train_test_split_demo(x, y, degrees, ratios, seed, shuffle):
    """polynomial regression with different split ratios and different degrees.""" 
    
    for ratio in ratios:
        for degree in degrees:      
            # split the data, and return train and test data
            data_train, data_test = split_data(x, y, ratio, seed, shuffle)
            
            # train data with polynomial basis
            x_train, y_train = data_train
            x_train = build_poly(x_train, degree)

            # test data with polynomial basis
            x_test, y_test = data_test
            x_test = build_poly(x_test, degree)

            # least squares
            _, weights = least_squares(y_train, x_train)

            # calculate RMSE 
            rmse_train = cts.compute_rmse(y_train, x_train, weights)
            rmse_test = cts.compute_rmse(y_test, x_test, weights)

            # print the results
            print("proportion={p}, degree={d}, Training RMSE={tr:.3f}, Testing RMSE={te:.3f}".format(
                  p=ratio, d=degree, tr=rmse_train, te=rmse_test))
