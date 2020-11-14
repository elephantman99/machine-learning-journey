# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    # this function should return the matrix formed
    # by applying the polynomial basis to the input data
    N = len(x)
    tx = np.zeros((N, degree+1))
    for j in range(degree+1):
        tx[:, j] = np.power(x, j)
    
    return tx