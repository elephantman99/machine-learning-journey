# -*- coding: utf-8 -*-
"""Function used to compute the loss."""

import numpy as np

def compute_loss(y, tx, w):
    """Calculate the loss using Mean Square Error.
   
    """
    N = len(y)
    e = y - np.dot(tx, w)
    return sum(e ** 2) / (2 * N)

def compute_mse(y, tx, w):
    """Calculate the loss using Mean Square Error.
   
    """
    
    N = len(y)
    e = y - tx @ w
    return sum(e ** 2) / (2 * N)
    

def compute_mae(y, tx, w):
    """Calculate the loss using Mean Absolute Error.
   
    """
    N = float(len(y))
    e = y - tx @ w
    return sum(np.absolute(e)) / (2 * N)
   
def compute_rmse(y, tx, w):
    """Calculate the loss using Root Mean Square Error.
   
    """
    return np.sqrt(2 * compute_mse(y, tx, w))


def ridge_mse(y, tx, ws, lambda_):
    """Calculate the loss using Ridge Mean Square Error.   
    """
    return compute_mse(y, tx, ws) + lambda_ * np.sum(ws ** 2)
    